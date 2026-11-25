"""Verify both transfer hashes and extract only regular files inside the result folder."""
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import tarfile

root=Path(__file__).resolve().parent/'outputs/pilot_20261120'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def unpack(archive,destination):
    destination.mkdir(parents=True,exist_ok=True)
    count=0
    with tarfile.open(archive,'r:gz') as stream:
        for member in stream:
            name=PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or ':' in member.name or '\\' in member.name:
                raise ValueError('Unsafe archive path')
            target=(destination/Path(*name.parts)).resolve()
            if not target.is_relative_to(destination.resolve()):raise ValueError('Archive outside destination')
            # Windows' default path limit can be shorter than a preserved run path.
            target=Path('\\\\?\\'+str(target))
            if member.isdir():target.mkdir(parents=True,exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True,exist_ok=True)
                with stream.extractfile(member) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
                count+=1
            else:raise ValueError('Unexpected archive member type')
    return count
outer=root/'pilot-delivery-20261120.tar.gz'
assert sha(outer)=='0acf872597784e83bca14d37f587facfd8bfdb95fb895aff94231de7b2d0a802'
unpack(outer,root/'delivery')
inner=root/'delivery/pilot-artifacts-20261120.tar.gz'
assert sha(inner)=='425e68bdecaa2840dc0de74d423bd52e99dd621b563a7fb2113aeb63fb01ca93'
count=unpack(inner,root/'raw')
shutil.copy2(root/'delivery/pilot-vm-description.json',root/'raw/pilot-vm-description.json')
print('Both archive SHA-256 checks passed;',count,'raw files extracted safely.')
