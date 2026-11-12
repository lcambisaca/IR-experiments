"""Shared-caption ablation with exact shared-record ranking and bounded peer revision."""
from dataclasses import asdict
import json
from pathlib import Path
import time

from .cache import digest, read_json
from .live import save_json, signature, generated_query_method, load_candidate_captions, resolve_target_image
from .multi_agent import RetrievalContext, question_key, valid_question, unique_proposals
from .shared_prompts import VERSION, CATEGORIES, messages
from .retrieval import ranks_for_targets

MODES = ('shared_neutral_three', 'shared_roles', 'shared_roles_critique')

def record_query(record):
    """Serialize actual observations verbatim; no generated summary can omit them."""
    return record['initial_caption'] + ''.join(
        ' Question: ' + qa['question'] + ' Answer: ' + qa['answer'] for qa in record['dialogue'])

def guard_tokens(retriever, query):
    tokenizer = retriever.processor.tokenizer
    limit = min(tokenizer.model_max_length,
                retriever.model.text_encoder.config.max_position_embeddings)
    count = len(tokenizer.encode(query, add_special_tokens=True, truncation=False))
    if count > limit:
        raise ValueError(f'Confirmed record requires {count} tokens, encoder limit {limit}; refusing truncation.')

class GuardedContext(RetrievalContext):
    def scores(self, query):
        guard_tokens(self.retriever, query)
        return super().scores(query)

    def select(self, neutral, proposals, interaction):
        for proposal in proposals:
            guard_tokens(self.retriever, neutral + ', ' + proposal['question'])
        return super().select(neutral, proposals, interaction)

class SharedChat:
    def __init__(self, transport, settings):
        self.transport, self.settings = transport, settings

    def begin(self, record, interaction):
        self.calls, self.audit = 0, []
        self.context = signature({'record': record, 'interaction': interaction})
        self.transport.begin_round(interaction, self.context)

    def complete(self, stage, role, payload):
        if self.calls >= self.settings.max_calls_per_round:
            raise RuntimeError('Shared-evidence round call budget exhausted.')
        self.calls += 1
        response = self.transport.create(messages=messages(stage, role, payload), model='local-qwen',
            n=1, max_tokens=512, temperature=.5 if stage == 'propose' else 0.,
            seed=int(signature([self.settings.seed, self.context, stage, role, payload])[:8], 16) % (2**31))
        choice = response['choices'][0]
        text = choice['message']['content']
        self.audit.append({'stage': stage, 'role': role, 'payload': payload, 'response': text})
        if choice.get('finish_reason') == 'length':
            raise ValueError('Shared JSON response truncated at explicit output cap.')
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError('Expected one JSON object.')
        return result

def validate_groups(result, candidates):
    """Enforce complete nonoverlapping IDs and literal provenance, not semantic truth."""
    captions = {c['id']: c['caption'] for c in candidates}
    seen = []
    groups, unknown = result.get('groups'), result.get('unknown_ids')
    if not isinstance(groups, list) or not isinstance(unknown, list):
        raise ValueError('Groups and unknown_ids must be lists.')
    values = set()
    for group in groups:
        value, ids, quotes = group.get('value'), group.get('ids'), group.get('quotes')
        if not isinstance(value, str) or not value.strip() or value.casefold() in values:
            raise ValueError('Group labels must be distinct nonempty values.')
        values.add(value.casefold())
        if not isinstance(ids, list) or not ids or not isinstance(quotes, dict):
            raise ValueError('Groups require IDs and literal quotes.')
        if set(quotes) != {str(i) for i in ids}:
            raise ValueError('Quote IDs do not match group IDs.')
        for i in ids:
            quote = quotes[str(i)]
            if (type(i) is not int or i not in captions or not isinstance(quote, str)
                    or not quote or quote not in captions[i] or value.casefold() not in quote.casefold()):
                raise ValueError('Unsupported candidate ID/value/quote.')
        seen.extend(ids)
    seen.extend(unknown)
    if any(type(i) is not int for i in seen) or len(seen) != len(set(seen)) or set(seen) != set(captions):
        raise ValueError('Each shared candidate ID must occur exactly once.')
    return {'groups': groups, 'unknown_ids': unknown,
            'explicit_coverage': (len(seen)-len(unknown))/max(1, len(seen))}

def gate(chat, record, proposal, candidates, role):
    q = proposal.get('question')
    if not valid_question(q):
        return {'status': 'malformed', 'reason': 'Expected one visual question.'}
    if question_key(q) in {question_key(x['question']) for x in record['dialogue']}:
        return {'status': 'already_asked', 'reason': 'Exact question already asked.'}
    quote = proposal.get('record_quote')
    if (proposal.get('category') not in CATEGORIES[role] or not isinstance(quote, str)
            or not quote.strip() or quote not in record_query(record)
            or not isinstance(proposal.get('unknown_detail'), str) or not proposal['unknown_detail'].strip()):
        return {'status': 'unsupported', 'reason': 'Invalid role category, unknown detail or record quote.'}
    answer = chat.complete('answerability', 'full', {'record': record, 'question': q}).get('answer')
    if not isinstance(answer, str) or not answer.strip():
        return {'status': 'malformed', 'reason': 'Missing record-only answerability response.'}
    if answer.strip().casefold() != 'uncertain':
        return {'status': 'already_known', 'reason': 'Record-only answer: ' + answer}
    check = chat.complete('assumptions', 'full', {'record': record, 'question': q})
    assumptions = check.get('unsupported_assumptions')
    if not isinstance(assumptions, list) or any(not isinstance(a, str) or not a.strip() for a in assumptions):
        return {'status': 'malformed', 'reason': 'Invalid presupposition audit.'}
    if assumptions:
        return {'status': 'unsupported', 'reason': '; '.join(assumptions)}
    grouping = chat.complete('group', role, {'proposal': proposal, 'candidates': candidates})
    try:
        grouping = validate_groups(grouping, candidates)
    except ValueError as error:
        return {'status': 'bad_grouping', 'reason': str(error), 'raw_grouping': grouping}
    return {'status': 'unknown', 'reason': check.get('reason', ''), 'grouping': grouping}

def question_round(chat, context, record, neutral, interaction, mode, shared):
    candidates = shared['captions']
    attempts, eligible, reviews = [], [], []
    roles = ['full']*3 if mode == MODES[0] else list(CATEGORIES)
    for slot, role in enumerate(roles):
        for attempt in range(chat.settings.replacement_attempts+1):
            p = {}
            payload = {'record': record, 'candidates': candidates, 'slot': slot,
                       'avoid': [a['proposal'].get('question') for a in attempts],
                       'previous_rejections': [a['filter'] for a in attempts if a['filter']['status'] != 'unknown']}
            try:
                p = chat.complete('propose', role, payload)
                p = {**p, 'owner': role, 'slot': slot}
                verdict = gate(chat, record, p, candidates, role)
            except (ValueError, TypeError, KeyError) as error:
                verdict = {'status': 'malformed', 'reason': str(error)}
            attempts.append({'role': role, 'slot': slot, 'attempt': attempt, 'proposal': p, 'filter': verdict})
            if verdict['status'] == 'unknown':
                eligible.append({**p, 'grouping': verdict['grouping']})
                break
    before = unique_proposals(eligible)
    pre, _ = context.select(neutral, before, interaction)
    final = before
    if mode == MODES[2]:
        final = []
        for p in before:
            reviewer = list(CATEGORIES)[(list(CATEGORIES).index(p['owner'])+1)%3]
            review = chat.complete('critique', reviewer, {'record': record, 'candidates': candidates, 'proposal': p})
            response = chat.complete('revise', p['owner'], {'record': record, 'candidates': candidates,
                                                         'proposal': p, 'critique': review})
            action = response.get('action')
            if action == 'withdraw':
                after = {'status': 'withdrawn', 'reason': response.get('reason', '')}
            elif action == 'retain' and response.get('question') == p['question']:
                after = {'status': 'unknown', 'reason': 'Retained previously validated question.', 'grouping': p['grouping']}
                final.append(p)
            elif action == 'revise':
                revised = {**response, 'owner': p['owner'], 'slot': p['slot']}
                after = gate(chat, record, revised, candidates, p['owner'])
                if after['status'] == 'unknown':
                    final.append({**revised, 'grouping': after['grouping']})
            else:
                after = {'status': 'malformed', 'reason': 'Invalid revision action.'}
            reviews.append({'before': p, 'reviewer': reviewer, 'critique': review, 'response': response, 'after_filter': after})
    final = unique_proposals(final)
    selected, kl = context.select(neutral, final, interaction)
    return {'shared': shared, 'attempts': attempts, 'before_critique': before, 'reviews': reviews,
            'final_proposals': final, 'pre_critique_selection': pre,
            'selected_question': selected, 'kl': kl, 'selection_changed': pre != selected}

def generate_shared_dialogue(retriever, *, initial_caption, target_id, image_path, context,
                             questioner, answerer, settings, n_states, checkpoint_path, run_identity, mode,
                             shared_cache):
    identity = signature([run_identity, initial_caption, target_id, digest(image_path), mode])
    path = Path(checkpoint_path)
    chat = SharedChat(questioner, settings)
    def rank(query):
        guard_tokens(retriever, query)
        return int(ranks_for_targets(retriever.scores([query]), [target_id])[0])
    if path.is_file():
        state = read_json(path)
        if state['identity'] != identity:
            raise ValueError('Shared checkpoint identity mismatch.')
    else:
        state = {'identity': identity, 'dialogue': [], 'states': [{'round': 0, 'query': initial_caption,
                 'raw_query': initial_caption, 'target_rank': rank(initial_caption)}], 'status': 'partial'}
        save_json(path, state)
    try:
        for interaction in range(len(state['states'])-1, n_states-1):
            record = {'initial_caption': initial_caption, 'dialogue': state['dialogue']}
            neutral = record_query(record)
            guard_tokens(retriever, neutral)
            key = signature([record, interaction])
            if key not in shared_cache:
                evidence = context.candidates(neutral, interaction, 'shared', record)
                evidence['captions'] = list({c['id']: c for c in evidence['captions']}.values())
                shared_cache[key] = evidence
            chat.begin(record, interaction)
            start = len(questioner.records)
            audit = question_round(chat, context, record, neutral, interaction, mode, shared_cache[key])
            q, answer = audit['selected_question'], None
            answer_start = len(getattr(answerer, 'records', []))
            dialogue = list(state['dialogue'])
            if q is not None:
                guard_tokens(retriever, neutral + '. ' + q)
                questioner.park()
                try:
                    answer = answerer.answer_original(context.original_answer, image_path, neutral + '. ' + q)
                finally:
                    answerer.park()
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError('Missing target answer.')
                dialogue.append({'question': q, 'answer': answer.strip()})
            query = record_query({'initial_caption': initial_caption, 'dialogue': dialogue})
            measured = rank(query)
            state['dialogue'] = dialogue
            state['states'].append({'round': interaction+1, 'query': query, 'raw_query': query,
                'target_rank': measured, 'question': q, 'answer': answer, 'audit': audit,
                'llm_calls': chat.calls, 'llm_audit': chat.audit, 'transport_records': questioner.records[start:],
                'answerer_records': getattr(answerer, 'records', [])[answer_start:]})
            state['status'] = 'partial'
            state.pop('last_error', None)
            state.pop('partial_round_audit', None)
            save_json(path, state)
            print(f'{mode} state {interaction+1}: {q!r}; answer={answer!r}; rank={measured}', flush=True)
        state['status'] = 'complete'
        save_json(path, state)
        return {**state, 'states': state['states'][:n_states]}
    except BaseException as error:
        state.update(status='interrupted', last_error=str(error), partial_round_audit=getattr(chat, 'audit', []))
        save_json(path, state)
        raise
    finally:
        questioner.park()
        answerer.park()

def prepare_shared_experiment(root, retriever, pairs, *, n_states, settings, device='cuda', offline=True):
    import importlib.metadata
    import torch
    from .evaluation import PreparedExperiment
    from .live_models import Qwen3, Blip2Answerer, require_cuda, QWEN_ID, QWEN_REVISION, BLIP2_ID, BLIP2_REVISION
    from .plugir_upstream import provenance
    require_cuda(device)
    settings.validate(len(retriever.cache.paths), n_states)
    if not pairs or len(pairs) > settings.max_targets:
        raise ValueError('Explicit development target budget exceeded.')
    root = Path(root)
    catalogue_path = root/'assets/protocol/visdial_captions.json'
    captions = load_candidate_captions(catalogue_path, retriever.cache.paths)
    identity = {'experiment': VERSION, 'settings': asdict(settings), 'cache': retriever.cache.report,
        'catalogue_sha256': digest(catalogue_path), 'upstream': provenance(),
        'qwen': [QWEN_ID,QWEN_REVISION], 'answerer': [BLIP2_ID,BLIP2_REVISION],
        'generation_device': device, 'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda,
        'versions': {name:importlib.metadata.version(name) for name in ('torch','transformers','fast-pytorch-kmeans')},
        'query_policy': 'verbatim original caption and actual QA; no generative summary; refuse truncation',
        'source_sha256': {p: digest(root/p) for p in ('src/agentir/shared_prompts.py', 'src/agentir/shared_evidence.py',
            'src/agentir/multi_agent.py','src/agentir/live_models.py','src/agentir/retrieval.py', 'run_shared_development.py')}}
    folder = root/'runs/shared_evidence'/signature(identity)[:20]
    save_json(folder/'identity.json', identity)
    questioner = Qwen3(folder/'qwen_calls', device=device, offline=offline, max_input_tokens=8192, seed=settings.seed)
    answerer = Blip2Answerer(folder/'blip2_calls', device=device, offline=offline)
    context = GuardedContext(retriever, captions, settings)
    conversations, elapsed, shared_cache = {}, {}, {}
    ids = {name:i for i,name in enumerate(retriever.cache.paths)}
    for mode in MODES:
        started = time.perf_counter()
        conversations[mode] = []
        for raw, _ in pairs:
            image = resolve_target_image(root/'data/images', raw['img'], download=not offline)
            conversations[mode].append(generate_shared_dialogue(retriever, initial_caption=raw['dialog'][0],
                target_id=ids[raw['img']], image_path=image, context=context, questioner=questioner,
                answerer=answerer, settings=settings, n_states=n_states,
                checkpoint_path=folder/mode/(signature(raw['img'])[:20]+'.json'),
                run_identity=signature(identity), mode=mode, shared_cache=shared_cache))
        elapsed[mode] = time.perf_counter()-started
    save_json(folder/'dialogues.json', conversations)
    diagnostics = []
    for mode, cases in conversations.items():
        for case_index, case in enumerate(cases):
            for state in case['states'][1:]:
                audit = state['audit']
                transport = state['transport_records']
                fresh = [r for r in transport if not r.get('reused', False)]
                diagnostics.append({'method':mode,'case_index':case_index,'round':state['round'],
                    'shared_caption_count':len(audit['shared']['captions']),
                    'proposal_attempts':len(audit['attempts']),
                    'rejected_attempts':sum(a['filter']['status']!='unknown' for a in audit['attempts']),
                    'unique_questions_before':len(audit['before_critique']),
                    'distinct_categories_before':len({p['category'] for p in audit['before_critique']}),
                    'eligible_questions_after':len(audit['final_proposals']),
                    'revisions_changed_question':sum(r['before']['question']!=r['response'].get('question')
                                                     for r in audit['reviews']),
                    'selection_changed':audit['selection_changed'],
                    'selected_question':state['question'],'target_rank':state['target_rank'],
                    'llm_calls':state['llm_calls'],'reused_calls':sum(bool(r.get('reused')) for r in transport),
                    'new_input_tokens':sum(r.get('input_tokens',0) for r in fresh),
                    'new_output_tokens':sum(r.get('output_tokens',0) for r in fresh),
                    'new_generation_seconds':sum(r.get('seconds',0.) for r in fresh)})
    save_json(folder/'diagnostics.json', diagnostics)
    if diagnostics:
        import csv
        with (folder/'diagnostics.csv').open('w',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(diagnostics[0]))
            writer.writeheader()
            writer.writerows(diagnostics)
    return PreparedExperiment(methods={m:generated_query_method(c) for m,c in conversations.items()},
        metadata={'identity':identity,'checkpoint_folder':str(folder),'wall_seconds':elapsed,
            'limitations':'Caption semantics/LLM filters remain fallible; literal quotes do not prove correct binding. '
            'KL is stability, not information gain. Representatives are not equal-probability targets. '
            'Critique uses more compute. New query policy means old runs are not an isolated role ablation.'},
        artifacts={'shared_dialogues.json':conversations,'shared_diagnostics.json':diagnostics})
