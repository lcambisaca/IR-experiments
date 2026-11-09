# Example conversations

Ranks are out of 50,000 photos (1 = top). "mix" = the rank the user would see with the caption mix.


## Drift: the microwave is outdoors on grass, but the rewrite moves it into a kitchen

Target: `unlabeled2017/000000239836.jpg` (photo: `figures/ex1_microwave.png`)


### PlugIR (live, Qwen)

Caption: *an older brown and black microwave that is smoking* (rank 1)

| Round | Question | BLIP-2 answer | Rank | Rank with mix |
|---|---|---|---|---|
| 1 | What is the location of the older brown and black microwave that is smoking? | a microwave oven | 1 | 1 |
| 2 | What is the context or activity happening near the smoking microwave? | a black and white microwave oven | 10 | 4 |
| 3 | What color are the cabinets in the kitchen where the older brown and black microwave is smoking? | a brown and black microwave oven | 46 | 9 |
| 4 | Is the smoke coming from the microwave due to a malfunction or is it normal operation? | a microwave oven | 18 | 4 |
| 5 | what is the other object in the image? | a brown and black microwave oven | 24 | 6 |
| 6 | What is the likely cause of the microwave smoking? | a malfunctioning microwave oven | 69 | 12 |
| 7 | What color are the kitchen cabinets in the scene where the older brown and black microwave is smoking? | white | 71 | 7 |
| 8 | What type of appliance is nearby the smoking microwave in the kitchen? | a microwave oven | 189 | 21 |
| 9 | What time of day is likely causing the microwave to smoke? | a microwave oven | 140 | 16 |
| 10 | What sound might be heard when the older brown and black microwave is smoking? | a scream | 100 | 8 |

Final description: *an older brown and black microwave oven smoking in a kitchen with white cabinets, likely due to a malfunction, surrounded by other microwave ovens, with a scream heard, during an unspecified time of day*


### Agent v2

Caption: *an older brown and black microwave that is smoking* (rank 1)

| Round | Question | BLIP-2 answer | Rank | Rank with mix |
|---|---|---|---|---|
| 1 | What is the exact color of the microwave (other than brown and black) that is smoking in the scene? | a black microwave | 1 | 1 |
| 2 | Is the smoke coming from the microwave due to overheating or a malfunction? | a black microwave oven | 1 | 1 |
| 3 | What type of kitchen is the smoking microwave located in? | a black microwave | 1 | 1 |
| 4 | Question: In which of the retrieval candidates is the smoking microwave located in a kitchen with cabinets and other standard kitchen elements? | a black microwave | 60 | 10 |
| 5 | What is the exact color of the microwave (other than brown and black) that is smoking in the scene? | a black microwave | 24 | 2 |
| 6 | In which of the retrieval candidates is the smoking microwave located in a kitchen with cabinets, countertops, and other standard kitchen elements such as a stove or? | a black microwave | 195 | 27 |
| 7 | What is the exact color of the microwave (other than brown and black) that is smoking in the scene? | a black microwave | 15 | 4 |
| 8 | What is the likely cause of the microwave smoking, and is it safe to use it after the incident? | a black microwave emitting smoke in a kitchen, with cabinets, counters, and typical | 14 | 2 |
| 9 | What is the exact location of the smoking microwave within the kitchen — for example, on a wooden bracket, a table, or directly on the countertop? | a black microwave emitting smoke is located in a kitchen with cabinets, countertops, and standard | 44 | 8 |
| 10 | what is the other object in the image? | a black microwave emits heat and smoke | 216 | 28 |

Final description: *a black microwave emitting smoke is located on a kitchen countertop, surrounded by cabinets and standard kitchen elements such as countertops and a stove, with no other prominent objects in the scene*


## Agent keeps the target; PlugIR's rewrite invents a second toilet

Target: `unlabeled2017/000000222337.jpg` (photo: `figures/ex2_toilet.png`)


### PlugIR (live, Qwen)

Caption: *a white toilet attached to a wooden wall sits in a tiled bathroom* (rank 1)

| Round | Question | BLIP-2 answer | Rank | Rank with mix |
|---|---|---|---|---|
| 1 | What is the condition of the toilet lid in the tiled bathroom? | a white toilet lid in a tiled bathroom | 1 | 1 |
| 2 | What color is the wall behind the white toilet in the tiled bathroom? | white | 1 | 1 |
| 3 | What color is the floor of the tiled bathroom with the white toilet? | white | 1 | 1 |
| 4 | What color is the tile flooring in the bathroom with the white toilet? | white | 1 | 1 |
| 5 | What type of bathroom fixture is located beside the white toilet in the tiled bathroom? | a toilet | 1 | 1 |
| 6 | What type of fixture is located beside the white toilet in the tiled bathroom? | a toilet | 17 | 2 |
| 7 | What type of fixture is located beside the white toilet in the tiled bathroom? | a white toilet with a white toilet lid sits in a white tiled bathroom alongside | 374 | 35 |
| 8 | What type of fixture is located beside the white toilet in the tiled bathroom? | a white toilet with a white toilet lid sits in a white tiled bathroom | 349 | 29 |
| 9 | What color is the sink in the tiled bathroom with the white toilet? | white toilet | 772 | 90 |
| 10 | what is the other object in the image? | a white toilet with a white toilet seat and lid | 916 | 164 |

Final description: *a white toilet with a white toilet lid sits in a white tiled bathroom alongside another white toilet, both with white sink fixtures and white floor tiles*


### Agent v2

Caption: *a white toilet attached to a wooden wall sits in a tiled bathroom* (rank 1)

| Round | Question | BLIP-2 answer | Rank | Rank with mix |
|---|---|---|---|---|
| 1 | What type of bathroom fixture is adjacent to the white toilet attached to a wooden wall? | a toilet | 1 | 1 |
| 2 | What is the shape or style of the wooden wall to which the white toilet is attached? | a white toilet attached to a wooden wall sits in a tiled bathroom with another | 2 | 1 |
| 3 | What is the color of the wooden wall to which the white toilet is attached? | white | 1 | 1 |
| 4 | What is the color of the tiles in the bathroom where the white toilet is attached to a wooden wall? | white | 1 | 1 |
| 5 | What is the color of the tiles in the bathroom where the white toilet is attached to a wooden wall? | white | 1 | 1 |
| 6 | What is the color of the tiles in the bathroom where the white toilet is attached to a wooden wall? | white | 1 | 1 |
| 7 | What is the color of the tiles in the bathroom where the white toilet is attached to a wooden wall? | white | 3 | 1 |
| 8 | What is the color of the tiles in the bathroom where the white toilet is attached to a wooden wall? | white | 1 | 1 |
| 9 | What is the color of the tiles in the bathroom where the white toilet is attached to a wooden wall? | white | 1 | 1 |
| 10 | What is the color of the tiles in the bathroom where the white toilet is attached to a wooden wall? | white | 1 | 1 |

Final description: *a white toilet attached to a white wooden wall sits in a bathroom with white tiles*


## Agent loses the target at round 4: BLIP-2 says the background is "white" (it is a street)

Target: `unlabeled2017/000000358981.jpg` (photo: `figures/ex3_apples.png`)


### PlugIR (live, Qwen)

Caption: *a man stands near some fruit on a sidewalk* (rank 2)

| Round | Question | BLIP-2 answer | Rank | Rank with mix |
|---|---|---|---|---|
| 1 | What type of fruit is the man near on the sidewalk? | apples | 1 | 1 |
| 2 | What is the man's activity while standing near the apples on the sidewalk? | he is standing near the apples on the sidewalk | 1 | 1 |
| 3 | What is the man's clothing or appearance while standing near the apples on the sidewalk? | a man stands near a pile of apples on a sidewalk | 1 | 1 |
| 4 | What color is the man's tie when he is standing near the apples on the sidewalk? | blue | 1 | 1 |
| 5 | What color is the man's jacket when he is standing near the apples on the sidewalk? | blue | 1 | 1 |
| 6 | what is the other object in the image? | a man in a blue jacket and blue tie stands near a pile of apples on | 1 | 1 |
| 7 | What color is the man's suit when he is standing near the apples on the sidewalk? | blue | 1 | 1 |
| 8 | What time of day is it when the man is standing near the apples on the sidewalk? | a man in a blue suit, blue jacket, and blue tie stands near a pile | 1 | 1 |
| 9 | What is the specific type of market or setting where the man is standing near the apples on the sidewalk? | a man in a blue suit, blue jacket, and blue tie stands near a pile | 6 | 1 |
| 10 | What is the man doing while standing near the apples on the sidewalk? | he is standing near the pile of apples on the sidewalk | 6 | 2 |

Final description: *a man in a blue suit, blue jacket, and blue tie stands near a pile of apples on a sidewalk in a market setting, during the day*


### Agent v2

Caption: *a man stands near some fruit on a sidewalk* (rank 2)

| Round | Question | BLIP-2 answer | Rank | Rank with mix |
|---|---|---|---|---|
| 1 | What specific type of fruit is the man standing near? | apples | 1 | 1 |
| 2 | What color is the man's clothing? | white | 1 | 1 |
| 3 | What type of market or setting is the man standing near the fruit in? | a market | 1 | 1 |
| 4 | What color is the background or setting of the fruit display? | white | 35 | 9 |
| 5 | What type of vegetable is the man standing near in the market? | a man in white clothing stands near a display of apples at a market | 36 | 8 |
| 6 | What type of vegetable is the man standing near in the market? | a man in white clothing stands near a display of apples at a market | 45 | 13 |
| 7 | What specific type of fruit is the man standing near in the market? | a man in white clothing stands near a display of apples at a market | 41 | 10 |
| 8 | What specific type of apples is the man standing near in the market? | a variety of apples | 43 | 12 |
| 9 | What specific type of apples is the man standing near in the market? | a variety of apples | 37 | 10 |
| 10 | What specific type of apples is the man standing near in the market? | apples | 43 | 12 |

Final description: *a man in white clothing stands near a variety of apples at a market, set against a white background*
