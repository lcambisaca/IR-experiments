# Conversations where the agent ended worse than PlugIR

Overnight run, first 30 targets, 10 rounds, same Qwen / BLIP-2 / BLIP for both. Rank = where the target photo sits among 50,000 after that round (1 is best; top 10 counts as found).

Agent options: each candidate question with Qwen's guessed answers and its split score (higher = the guessed answers divide the current candidates more evenly; ★ = asked).

## 1. [000000185565.jpg](http://images.cocodataset.org/unlabeled2017/000000185565.jpg): PlugIR rank 1, agent rank 15

**Caption:** a bedroom is filled with lots of posters and a busy computer desk  
**Rank before any questions:** 1

| Round | PlugIR: question → answer | Rank | Agent: question → answer | Rank | Agent's options (guessed answers, score) |
|---|---|---|---|---|---|
| 1 | What types of posters are displayed on the walls of the bedroom? → *posters* | 1 | What specific items, besides posters, are on the computer desk in the bedroom? → *a computer monitor* | 1 | posters, laptop, keyboard (0.34)<br>★ laptop, printer, keyboard (0.68)<br>laptop, printer, keyboard (0.56) |
| 2 | What specific items are on the computer desk in the bedroom? → *a laptop computer* | 1 | What color is the bed in the bedroom? → *white* | 1 | ★ blue bed, red bed, green bed (0.65) |
| 3 | What types of content are featured on the posters in the bedroom? → *a bedroom filled with posters and a busy computer desk featuring a laptop computer* | 1 | What color are the posters in the bedroom? → *white* | 3 | ★ red, blue, yellow (0.80)<br>laptop, monitor, computer (0.36) |
| 4 | What other electronic devices are on the computer desk in the bedroom besides the laptop? → *a television* | 1 | What type of furniture is in the bedroom besides the bed and the computer desk? → *a desk* | 10 | ★ white posters, cluttered desk, vibrant atmosphere (0.24) |
| 5 | What colors or themes do the posters on the walls have in the bedroom? → *a bedroom filled with posters featuring various content and a busy computer desk with a laptop and* | 2 | What color are the pillows on the bed in the bedroom? → *white* | 27 | ★ white pillows, white and tan, colorful blanket (0.56)<br>laptops, mouse, keyboard (0.19)<br>no personal items visible, only a computer and monitor (0.12)<br>no books or items visible, only a monitor and keyboard, no personal items shown (0.17) |
| 6 | What types of electronic devices are on the computer desk in the bedroom besides the laptop and television? → *a computer monitor* | 2 | What type of bed is in the bedroom? → *a white bed* | 13 | ★ white bed with pillows, double bed, king size bed (0.19) |
| 7 | What types of books or reading materials are present in the bedroom? → *a bedroom filled with colorful posters and a busy computer desk with a laptop, television,* | 1 | What other items, besides posters and the computer monitor, are on the computer desk in the bedroom? → *a white chair* | 10 | yes, there are books and posters, no personal items visible, books and posters present (0.14)<br>★ books, keyboard, mouse (0.25)<br>books, keyboard, mouse (0.16) |
| 8 | What specific topics or subjects are depicted in the posters on the walls of the bedroom? → *a bedroom adorned with posters of various themes* | 2 | What type of bed is in the bedroom? → *a white bed* | 7 | ★ king size bed, white bed, bed with pillows (0.15) |
| 9 | What specific themes or subjects are depicted in the posters on the walls of the bedroom? → *a bedroom filled with colorful posters and posters of various themes and subjects* | 2 | What is the overall style or theme of the posters on the wall in the bedroom? → *a white bedroom* | 34 | ★ abstract, vintage, cartoon (0.51)<br>white chair, wooden chair, office chair (0.35)<br>white chair, wooden chair, office chair (0.31) |
| 10 | What specific themes or subjects are depicted in the posters on the walls of the bedroom? → *a bedroom filled with colorful posters and posters of various themes and subjects* | 1 | What type of chair is located on the computer desk in the bedroom? → *a white chair* | 15 | busy and scattered, grouped together, mixed placement (0.04)<br>★ white chair, wooden chair, recliner chair (0.25)<br>white chair, office chair, recliner chair (0.11) |

**PlugIR final search:** A bedroom adorned with vibrant posters on various themes and topics, and a bustling computer desk equipped with a laptop, television, and computer monitor.  
**Agent final search:** a white bedroom filled with white posters and a busy computer desk featuring a computer monitor and a white chair, with white pillows on a white bed and a white chair on the desk, overall white-themed decor

## 2. [000000307868.jpg](http://images.cocodataset.org/unlabeled2017/000000307868.jpg): PlugIR rank 6, agent rank 54

**Caption:** a white river boat sales towards mountains in the background  
**Rank before any questions:** 9

| Round | PlugIR: question → answer | Rank | Agent: question → answer | Rank | Agent's options (guessed answers, score) |
|---|---|---|---|---|---|
| 1 | what is the other object in the image? → *a white river boat* | 5 | What is the activity of the people on the white river boat? → *a white river boat* | 5 | ★ selling boats, traveling, fishing (0.49)<br>river boat, white boat, boat on river (0.14)<br>toward, downstream, northward (0.26)<br>white (0.00) |
| 2 | what is the other object in the image? → *a white river boat sailing towards mountains in the background* | 5 | what is the other object in the image? → *a white river boat sailing towards mountains in the background* | 5 | white (0.00)<br>★ mountains, white boat, sailing vessel (0.37)<br>white (0.00)<br>white (0.00) |
| 3 | what is the other object in the image? → *a white riverboat* | 5 | What activity are the people on the boat engaged in? → *sailing* | 32 | no people on boat (0.00)<br>★ no activity, no people, empty boat (0.10)<br>no people on the boat (0.00) |
| 4 | what is the other object in the image? → *a white riverboat sailing towards mountains in the background* | 5 | What is the weather condition in the scene? → *a white river boat sailing towards mountains in the background* | 61 | ★ sunny, clear sky, blue sky (0.47)<br>white (0.00) |
| 5 | what is the other object in the image? → *a white riverboat* | 5 | **Question:** What is the color of the mountains in the background of the scene? → *white* | 8 | mountains, sailing people, clear sky (0.33)<br>green, blue, white (0.41)<br>towards the mountains (0.00)<br>★ blue, green, white (0.55)<br>green, blue, white (0.51) |
| 6 | What type of boat is sailing towards the mountains? → *a white riverboat* | 5 | **Question:** What direction is the white river boat traveling toward the mountains? → *a white river boat traveling towards the mountains* | 6 | morning, afternoon, evening (0.18)<br>towards the mountains (0.00)<br>★ toward the mountains, sailing toward, heading toward (0.22) |
| 7 | what is the other object in the image? → *a white riverboat* | 5 | What type of river is the boat traveling on? → *a river* | 70 | ★ freshwater river, mountain river, white river (0.24)<br>white (0.00)<br>white (0.00)<br>white (0.00) |
| 8 | what is the other object in the image? → *a white riverboat sailing towards mountains in the background* | 5 | What is the color of the river in the background of the scene? → *white* | 64 | morning, afternoon, evening (0.12)<br>★ clear, white, blue (0.32)<br>clear, white, blue (0.24) |
| 9 | What color is the boat’s lettering? → *a white riverboat heading toward mountains in the distance* | 5 | **Question:** What is the color of the mountains in the background of the scene? → *white* | 64 | morning, afternoon, evening (0.17)<br>white, gray, blue (0.10)<br>★ white, gray, blue (0.47)<br>white, snowy, gray (0.26) |
| 10 | What color is the lettering on the boat sailing towards the mountains? → *white* | 6 | What time of day is depicted in the scene with the white river boat sailing toward the mountains? → *a white river boat is sailing toward the mountains* | 54 | ★ morning light, golden hour, sunset glow (0.30)<br>(no usable guesses) (0.00)<br>river boat, white boat, sailing boat (0.23) |

**PlugIR final search:** a white riverboat with white lettering sailing towards mountains in the background  
**Agent final search:** a white river boat sailing towards white mountains in the background, with people engaged in sailing, during an unspecified time of day

## 3. [000000112857.jpg](http://images.cocodataset.org/unlabeled2017/000000112857.jpg): PlugIR rank 2, agent rank 17

**Caption:** plates and bowls of food are sitting on top of a table  
**Rank before any questions:** 35

| Round | PlugIR: question → answer | Rank | Agent: question → answer | Rank | Agent's options (guessed answers, score) |
|---|---|---|---|---|---|
| 1 | What types of food are specifically included in the bowls or plates on the table? → *rice* | 19 | What types of food are specifically on the white plate? → *fried rice* | 42 | ★ pizzas, vegetables, sandwiches (1.01)<br>vegetables, eggs, sausage (0.80)<br>vegetables, fruit, sandwiches (0.96) |
| 2 | What types of food are specifically included in the bowls or plates on the table besides rice? → *chicken* | 3 | What types of food are on the plate with carrots, potatoes, and chicken? → *fried rice* | 2 | fried rice, steak, broccoli (0.24)<br>fries, eggs, bacon (0.26)<br>★ fried rice, vegetables, protein (0.32)<br>fried rice, white plate, table (0.22) |
| 3 | What types of vegetables are included in the bowls or plates on the table? → *rice and vegetables* | 3 | What types of food are on the wooden table that do not include a white plate? → *fried rice, carrots, potatoes, and chicken* | 4 | ★ fried rice, carrots, potatoes (0.46)<br>sliced vegetables (0.00) |
| 4 | What type of vegetables are included in the bowls or plates on the table? → *a variety of vegetables* | 3 | What types of food are on the plates and bowls that are not on a white plate and do not include carrots, potatoes, or chicken? → *fried rice* | 3 | ★ steak, broccoli, fruit (0.37) |
| 5 | What specific vegetables are mentioned in the retrieval candidates beyond just "a variety of vegetables"? → *a variety of vegetables* | 4 | What types of food are on the plates and bowls that are on a wooden table and do not include fried rice, carrots, potatoes, or chicken? → *fried rice* | 6 | potatoes, chicken, veggies (0.06)<br>none available (0.00)<br>★ no food matches the criteria, only vegetables are present, no such plates exist (0.06) |
| 6 | What specific vegetables are mentioned in the retrieval candidates that are served with chicken? → *a table filled with plates of food, including rice, chicken, and vegetables* | 2 | what is the other object in the image? → *a plate of fried rice* | 2 | ★ carrots, potatoes, chicken (0.20)<br>fried rice, carrots, potatoes (0.14)<br>fried rice, carrots, potatoes (0.15) |
| 7 | What specific vegetables are served with chicken in the bowls or plates on the table? → *a table filled with plates of food, including rice, chicken, and vegetables* | 2 | What types of food are on the plates and bowls that are on a wooden table and do not include carrots, potatoes, or chicken? → *fried rice* | 2 | brown, dark, light (0.29)<br>★ tortilla, veggies, cheese, broccoli, rice, steak, meat, rice, condiments (0.38)<br>fried rice only, rice and meat, no food matches (0.09)<br>fried rice only, rice and meat, tortilla and cheese (0.26) |
| 8 | What specific vegetables are served with chicken in the bowls or plates on the table? → *a table filled with plates of food* | 2 | What types of food are on the plates and bowls that are on the table and do not include a white plate, carrots, potatoes, or chicken? → *fried rice* | 3 | fried rice, carrots, potatoes (0.16)<br>★ tortilla, veggies, cheese, french toast, hot dogs, fries, rice, broccoli, steak (0.57) |
| 9 | What specific vegetables are served with chicken in the bowls or plates on the table? → *a table laden with plates of food* | 2 | What types of food are on the plates and bowls that are on the table and do not include a white plate, fried rice, carrots, potatoes, or chicken? → *fried rice* | 5 | ★ none exist (0.00)<br>(fell back to PlugIR's choice) |
| 10 | What type of dish or container is used to hold the food on the table? → *a plate* | 2 | What types of food are on the plates and bowls that are on the table and do not include a white plate, carrots, potatoes, chicken, or fried rice? → *fried rice* | 17 | ★ salad, pizza, soda (0.10) |

**PlugIR final search:** A table laden with plates of food featuring rice, chicken, and a diverse array of vegetables.  
**Agent final search:** plates and bowls of food, including fried rice, carrots, potatoes, and chicken, are arranged on a wooden table, with a white plate containing fried rice and no other distinct food items, and no food present that excludes all listed items

## 4. [000000347725.jpg](http://images.cocodataset.org/unlabeled2017/000000347725.jpg): PlugIR rank 9, agent rank 73

**Caption:** 6 people are sitting around a table with 2 pizzas and drinks  
**Rank before any questions:** 2

| Round | PlugIR: question → answer | Rank | Agent: question → answer | Rank | Agent's options (guessed answers, score) |
|---|---|---|---|---|---|
| 1 | What type of pizza is being served at the table? → *pizza* | 1 | What color is the pizza being served at the table? → *red* | 7 | mixed gender, all male, all female (0.47)<br>pepperoni, cheese, margherita (0.51)<br>★ red, white, green (0.57)<br>wooden plates, plates not specified, metal plates (0.49)<br>beer, soda, water (0.54) |
| 2 | What colors are the pizzas? → *red* | 3 | What type of toppings are on the pizzas being served at the table? → *cheese* | 5 | large, two, big (0.10)<br>★ pepperoni, red sauce, no toppings (0.45)<br>red pizza, pepperoni pizza, pizza with red sauce (0.25) |
| 3 | What is the arrangement of the 6 people around the table? → *a group of 6 people are sitting around a table with two red pizzas and drinks* | 7 | What gender composition is among the six people sitting at the table? → *male* | 22 | ★ mixed genders, all male, all female (0.51)<br>around a table, sitting together, at a round table (0.17)<br>mixed, varied, different (0.04) |
| 4 | What is the specific topping on one of the pizzas? → *cheese* | 5 | What is the exact arrangement of the two pizzas on the table? → *a pizza with cheese and toppings* | 20 | water, soda, beer (0.36)<br>★ sitting around table, eating red pizzas, cheese toppings present (0.45)<br>eating, reaching, holding (0.27) |
| 5 | What is the exact number of slices on each pizza? → *6* | 4 | What are the drinks being served with the pizzas at the table? → *a beer* | 38 | no child in the image (0.00)<br>side by side, on the table, with cheese toppings (0.24)<br>nothing specific, having a meal, talking about pizza (0.19)<br>★ water, soda, beer (0.39)<br>all male, mixed genders, six men (0.17) |
| 6 | What is the gender composition of the six people sitting at the table? → *a man and a woman* | 2 | What is the specific position of the pizza slices on the pizzas at the table? → *a pizza with cheese and meat* | 62 | ★ half-cut slices, slices cut in half, slices shared equally (0.10)<br>half-cut slices, whole slices, no slices cut (0.08) |
| 7 | What is the gender composition of the other four people at the table besides the man and woman? → *a man and a woman* | 1 | What is the exact number of wine glasses present at the table? → *a few* | 66 | ★ none, zero, no wine glasses (0.22) |
| 8 | What is the exact number of slices on each pizza? → *a pizza* | 2 | What is the exact number of men among the six people sitting at the table? → *six* | 80 | no children present, only one child, two children present (0.05)<br>none, zero, no one (0.13)<br>adults only, mixed ages, all adults (0.07)<br>★ all six are men, five men and one woman, six men present (0.28) |
| 9 | What is the exact number of people at the table, and how are they distributed in terms of gender? → *a group of 6 people — two men and two women — are sitting around a* | 2 | What is the exact arrangement of the two pizzas on the table? → *a pizza with cheese and meat* | 99 | ★ slices on cheese and meat pizzas, on wooden plates, with beer and wine glasses (0.57)<br>wooden board, wooden plates, large board (0.28) |
| 10 | What type of wooden board is the large pizza placed on? → *a large pizza* | 9 | What is the exact number of women among the six people sitting at the table? → *no* | 73 | ★ zero, no women, no females (0.19) |

**PlugIR final search:** 6 people — two men and two women — are sitting around a table with two red pizzas, each cut into 6 slices, and drinks; one pizza has cheese topping, placed on a wooden board.  
**Agent final search:** Six men are sitting around a table with two pizzas—one with cheese and toppings, the other with cheese and meat—served alongside a few wine glasses and a beer.

## 5. [000000555125.jpg](http://images.cocodataset.org/unlabeled2017/000000555125.jpg): PlugIR rank 9, agent rank 65

**Caption:** a church steeple on a cloudy day with blue skies  
**Rank before any questions:** 14

| Round | PlugIR: question → answer | Rank | Agent: question → answer | Rank | Agent's options (guessed answers, score) |
|---|---|---|---|---|---|
| 1 | What color is the building that houses the church steeple? → *blue* | 19 | What color is the building that has the steeple in the scene? → *blue* | 11 | ★ red brick, brick, stone (0.51)<br>red brick (0.00)<br>red brick (0.00) |
| 2 | what is the other object in the image? → *a blue church with a steeple on a cloudy day* | 14 | What is the architectural style of the building with the steeple? → *a blue church steeple on a cloudy day with blue skies* | 12 | ★ gothic, victorian, romanesque (0.36)<br>blue church steeple, religious building, church tower (0.24)<br>pointed, spire, tower (0.09)<br>blue church steeple (0.00) |
| 3 | What color is the steeple of the church? → *a blue church with a steeple on a cloudy day* | 15 | What color is the brick structure in the scene? → *a blue church steeple visible on a cloudy day beneath blue skies.* | 12 | ★ brown brick structure (0.00)<br>(fell back to PlugIR's choice) |
| 4 | What is the color of the sky in the image? → *blue* | 20 | What type of structure is the building with the steeple in the scene? → *a church steeple* | 11 | blue steeple, cloudy sky, church tower (0.15)<br>church steeple, blue steeple, religious tower (0.07)<br>★ church, religious building, place of worship (0.20)<br>blue church steeple, church with steeple, steeple building (0.08)<br>church, religious building, place of worship (0.16) |
| 5 | what is the other object in the image? → *a blue church with a blue steeple on a blue sky* | 29 | What time of day is depicted in the scene with the church steeple and cloudy skies? → *a blue sky* | 12 | church tower, clock tower, steeple (0.18)<br>★ afternoon, evening, morning (0.31)<br>church tower, blue steeple, religious building (0.16)<br>church tower, steeple, religious structure (0.13) |
| 6 | What is the color of the surrounding trees or natural elements near the church? → *a blue church with a blue steeple on a blue sky* | 11 | What type of building has the steeple and is made of red brick? → *a church steeple* | 55 | ★ red brick building with steeple (0.00)<br>red brick building with steeple (0.00)<br>(fell back to PlugIR's choice) |
| 7 | What time of day is depicted in the image based on the cloud coverage and sky color? → *a blue church with a blue steeple on a blue sky, surrounded by trees* | 8 | what is the other object in the image? → *a blue church steeple on a cloudy day with blue sky and red brick structure in* | 56 | ★ blue steeple, red brick building, clock tower (0.47)<br>blue, cloudy, overcast (0.15)<br>blue, cloudy, overcast (0.12) |
| 8 | What color are the surrounding trees or natural elements near the church? → *a blue church with a blue steeple stands on a blue sky, surrounded by* | 9 | **Question:** What color is the red brick structure in the scene with the church steeple and cloudy skies? → *a blue church steeple on a cloudy day with blue sky and a red brick* | 58 | red brick structure (0.00)<br>blue, cloudy, clear (0.18)<br>red brick church, old brick building, brick church (0.15)<br>★ red, brown, bricks (0.24)<br>red, brown, bricks (0.24) |
| 9 | **Question:** What type of building is the church located in? → *a church* | 7 | what is the other object in the image? → *a blue church steeple on a cloudy day with blue sky and red brick structure* | 52 | blue church steeple, red brick church, church building (0.12)<br>★ blue steeple, cloudy sky, red brick structure (0.21)<br>red brick building with a steeple (0.00) |
| 10 | What type of building is the church located in? → *a blue church with a blue steeple standing on a blue sky* | 9 | What is the relationship between the church steeple and the red brick structure in the scene? → *a blue church steeple* | 65 | ★ church and building are together (0.00)<br>church and building are together (0.00)<br>(fell back to PlugIR's choice) |

**PlugIR final search:** a blue church with a blue steeple standing on a blue sky, surrounded by trees, on a cloudy day  
**Agent final search:** a blue church steeple on a cloudy day with blue skies and a red brick structure, where the steeple is part of the building, both situated in a serene, quiet landscape

## 6. [000000296319.jpg](http://images.cocodataset.org/unlabeled2017/000000296319.jpg): PlugIR rank 2, agent rank 11

**Caption:** a toilet sitting in the middle of an abandoned building  
**Rank before any questions:** 16

| Round | PlugIR: question → answer | Rank | Agent: question → answer | Rank | Agent's options (guessed answers, score) |
|---|---|---|---|---|---|
| 1 | What type of toilet is located in the middle of the abandoned building? → *a toilet* | 12 | What color is the plastic cistern associated with the toilet in the middle of the abandoned building? → *a toilet in the middle of an abandoned building* | 22 | white toilet, old toilet, rusted toilet (0.33)<br>white, blue, brown (0.64)<br>★ blue, green, yellow (0.71)<br>abandoned warehouse (0.00)<br>white, blue, brown (0.62) |
| 2 | What type of cistern is associated with the toilet in the middle of the abandoned building? → *a toilet* | 15 | What type of toilet lid is on the toilet in the middle of the abandoned building? → *a toilet with a plastic cistern* | 20 | ★ closed lid, open lid, no lid (0.10) |
| 3 | What type of cistern is associated with the toilet in the middle of the abandoned building? → *a toilet* | 12 | What is the condition of the walls surrounding the toilet in the abandoned building? → *a toilet with a plastic cistern sits in the middle of an abandoned building* | 24 | ★ crumbling, broken, old (0.18)<br>plastic cistern, no color specified, unknown color (0.14) |
| 4 | What is the condition of the toilet in the middle of the abandoned building? → *it is in the middle of the building* | 5 | what is the other object in the image? → *a toilet with a plastic cistern sits in the middle of an abandoned building* | 23 | middle of abandoned building, abandoned building interior, no specific location given (0.13)<br>★ broken wall, deteriorated floor, rusty pipe (0.45)<br>abandoned warehouse, ruined apartment, old factory building (0.29)<br>abandoned warehouse, ruined apartment, old factory building (0.32) |
| 5 | What color is the toilet in the middle of the abandoned building? → *white* | 8 | What is the color of the walls surrounding the toilet in the middle of the abandoned building? → *white* | 34 | ★ cracked, faded, brown (0.38)<br>cracked, faded, brown (0.35) |
| 6 | What type of flooring is beneath the white toilet in the middle of the abandoned building? → *stone* | 2 | What color is the seat of the toilet in the middle of the abandoned building? → *a toilet with a plastic seat sits in the middle of an abandoned building* | 69 | ★ red, white, blue (0.51) |
| 7 | What type of building is the toilet located in the middle of? → *a white toilet sits in the middle of an abandoned building* | 1 | What color is the toilet seat in the bathroom with a plastic cistern in the middle of an abandoned building? → *white* | 20 | ★ white, blue, brown (0.53) |
| 8 | What time of day is depicted in the scene of the white toilet in the middle of an abandoned building? → *a white toilet is located at the center of a deserted building* | 2 | what is the other object in the image? → *a toilet with a white plastic seat and cistern sits in the middle of* | 13 | ★ white cistern, white seat, abandoned building (0.19)<br>white seat (0.00)<br>white seat (0.00) |
| 9 | what is the other object in the image? → *a white toilet sits in the middle of an abandoned building on stone flooring* | 1 | What color is the toilet cistern in the image? → *white* | 20 | ★ white (0.00)<br>(fell back to PlugIR's choice) |
| 10 | what is the other object in the image? → *a white toilet* | 2 | what is the other object in the image? → *a toilet with a white plastic seat and cistern sits in the middle of* | 11 | ★ sink, oven, window (0.39)<br>white plastic (0.00)<br>white plastic (0.00)<br>white plastic (0.00) |

**PlugIR final search:** a white toilet sits in the middle of an abandoned building on stone flooring  
**Agent final search:** A toilet, featuring a white plastic tank and white seat, is located in the center of a deserted building with white walls.

## 7. [000000240413.jpg](http://images.cocodataset.org/unlabeled2017/000000240413.jpg): PlugIR rank 2, agent rank 9

**Caption:** in the downtown market place there are a bunch of people on bikes and moving around  
**Rank before any questions:** 1

| Round | PlugIR: question → answer | Rank | Agent: question → answer | Rank | Agent's options (guessed answers, score) |
|---|---|---|---|---|---|
| 1 | What type of bikes are the people riding in the downtown marketplace? → *a bicycle* | 1 | What color are the bicycles the people are riding in the downtown marketplace? → *red* | 1 | regular bikes, city bikes, recumbent bikes (0.07)<br>★ blue, red, green (0.64)<br>bicycles, riding bikes, people on bikes (0.18) |
| 2 | What color of bikes are the people riding in the downtown marketplace? → *a city street with people riding bikes* | 1 | What is the weather condition based on the presence of red umbrellas in the downtown marketplace? → *a city street* | 19 | bicycling in the market, shopping at the market, marketplace commute (0.38)<br>★ rainy, sunny, cloudy (0.45)<br>downtown business district, central market area, city center hub (0.29)<br>across the street, toward the market, through the area (0.23)<br>morning, afternoon, late afternoon (0.06) |
| 3 | What type of clothing is the man wearing while riding his bike in the downtown marketplace? → *a white shirt* | 1 | What type of traditional clothing are the two women wearing while riding bicycles in the downtown marketplace? → *a traditional outfit* | 15 | ★ traditional thai clothing (0.00)<br>(fell back to PlugIR's choice) |
| 4 | What type of market or environment is the downtown marketplace known for? → *a city street* | 1 | What type of market setting is described by the presence of people on bikes and red umbrellas? → *a market* | 11 | shopping, browsing, chatting (0.09)<br>★ downtown bicycle market, rainy city market, red umbrella market (0.38)<br>rainy weather, it's raining, wet streets (0.04)<br>downtown marketplace, rainy city street, urban market scene (0.31) |
| 5 | What type of people are riding the bikes in the downtown marketplace? → *a man in a white shirt* | 3 | What specific activities are the people on bikes engaged in within the downtown marketplace? → *a man riding a bike* | 20 | ★ riding bicycles, wearing traditional outfits, navigating busy street (0.31)<br>red (0.00)<br>red (0.00)<br>red bicycles (0.00) |
| 6 | What color shirt is the man wearing while riding his bike in the downtown marketplace? → *white shirt* | 3 | What is the gender of the person riding the bike with the dog in the downtown marketplace? → *a man* | 6 | ★ female (0.00)<br>female (0.00)<br>(fell back to PlugIR's choice) |
| 7 | What time of day is it in the downtown marketplace when people are riding bikes? → *a man wearing a white shirt cycles through a downtown marketplace* | 2 | What is the purpose or significance of the bicycles in the downtown marketplace setting? → *a mode of transportation* | 10 | ★ transportation, commute, local commerce (0.07)<br>dog (0.00)<br>dog (0.00) |
| 8 | What type of bikes are the people riding in the downtown marketplace? → *a bicycle* | 4 | what is the other object in the image? → *a man riding a bike* | 19 | ★ dog, umbrella, bike (0.23)<br>red bicycles (0.00) |
| 9 | What type of bike is the man in a white shirt riding in the downtown marketplace? → *a bicycle* | 2 | What type of market is specifically described by the presence of people on bikes and red umbrellas? → *a city market* | 11 | ★ downtown marketplace, rainy market, bike-friendly market (0.16) |
| 10 | What type of bike is the man in a white shirt riding in the downtown marketplace? → *a bicycle* | 2 | What time of day is it based on the lighting and activity in the downtown marketplace? → *a sunny day* | 9 | ★ morning, afternoon, evening (0.07)<br>red umbrellas (0.00) |

**PlugIR final search:** A man wearing a white shirt cycles through a bustling downtown marketplace on a city street, where pedestrians are also riding bicycles.  
**Agent final search:** In a sunny day downtown city market, people ride red bicycles as a mode of transportation, with a man riding a bike and a dog, two women in traditional outfits, and red umbrellas indicating the weather—all set in a bustling, active market environment.

## 8. [000000029737.jpg](http://images.cocodataset.org/unlabeled2017/000000029737.jpg): PlugIR rank 19, agent rank 76

**Caption:** a man puts his hand in the water as his body leans backwards on a white surfboard among the ocean waves  
**Rank before any questions:** 34

| Round | PlugIR: question → answer | Rank | Agent: question → answer | Rank | Agent's options (guessed answers, score) |
|---|---|---|---|---|---|
| 1 | What color is the wetsuit the man is wearing? → *white* | 13 | What color is the wetsuit the man is wearing when he puts his hand in the water and leans backwards? → *a man puts his hand in the water as his body leans backwards on a* | 69 | ★ black wetsuit (0.00)<br>white (0.00)<br>white surfboard (0.00)<br>(fell back to PlugIR's choice) |
| 2 | what is the other object in the image? → *a man in a white wetsuit lays on a white surfboard* | 17 | What is the texture or style of the white surfboard the man is using? → *a man in a blue wetsuit lays on a white surfboard* | 59 | ★ smooth, glossy, polished (0.03)<br>white (0.00)<br>white (0.00) |
| 3 | what is the other object in the image? → *a man in a white wetsuit leans backwards on a white surf* | 20 | what is the other object in the image? → *a man in a blue wetsuit* | 69 | ★ white surfboard (0.00)<br>white surfboard (0.00)<br>white surfboard (0.00)<br>(fell back to PlugIR's choice) |
| 4 | what is the other object in the image? → *a man in a white wetsuit leans backwards on a white surf* | 21 | what is the other object in the image? → *a man in a blue wetsuit lays on a white surfboard* | 75 | ★ white surfboard (0.00)<br>white surfboard (0.00)<br>white surfboard (0.00)<br>(fell back to PlugIR's choice) |
| 5 | What action is the man taking with his hand while leaning backwards on the surfboard? → *a man in a white wetsuit leans backwards on a white surf* | 16 | What is the direction of the ocean waves in relation to the man on the surfboard? → *a man in a blue wetsuit lays on the beach with his hands in the* | 65 | ★ breaking surface, rising toward, moving forward (0.05)<br>white (0.00) |
| 6 | what is the other object in the image? → *a man in a white wetsuit leans backwards on a white surf* | 15 | **Question:** What color is the surfboard the man is using when he puts his hand in the water and leans backwards? → *white* | 93 | no other objects visible (0.00)<br>blue wetsuit (0.00)<br>★ blue, white, not specified (0.12) |
| 7 | what is the other object in the image? → *a man in a white wetsuit leans backwards on a white surf* | 16 | What is the man’s facial expression as he leans backwards on the surfboard with his hand in the water? → *a man in a blue wetsuit leans backwards on a white surf* | 99 | ★ smiling, calm, focused (0.08)<br>blue wetsuit (0.00) |
| 8 | what is the other object in the image? → *a man in a white wetsuit leans backwards on a white surf* | 15 | What time of day is depicted in the scene with the man leaning backwards on the surfboard? → *a man in a blue wetsuit leans backwards on a white surf* | 165 | ★ morning, afternoon, evening (0.15)<br>blue wetsuit (0.00) |
| 9 | what is the other object in the image? → *a man in a white wetsuit* | 16 | what is the other object in the image? → *a man in a blue wetsuit leans backwards on a white surf* | 162 | ★ white surfboard (0.00)<br>blue wetsuit (0.00)<br>(fell back to PlugIR's choice) |
| 10 | what is the other object in the image? → *a man in a white wetsuit leans backwards on a white surf* | 19 | what is the other object in the image? → *a man in a blue wetsuit leans backwards on a white surf* | 76 | ★ no other objects visible (0.00)<br>blue wetsuit (0.00)<br>(fell back to PlugIR's choice) |

**PlugIR final search:** A man in a white wetsuit is leaning back on a white surfboard, his hand extended into the water amidst the ocean waves.  
**Agent final search:** a man in a blue wetsuit leans backwards on a white surfboard with his hand in the water, surrounded by ocean waves, facing the direction of the waves, with a neutral facial expression, during daytime, and no other prominent objects in the scene

## 9. [000000221035.jpg](http://images.cocodataset.org/unlabeled2017/000000221035.jpg): PlugIR rank 26, agent rank 101

**Caption:** lots of boats are docked in several straight rows  
**Rank before any questions:** 9

| Round | PlugIR: question → answer | Rank | Agent: question → answer | Rank | Agent's options (guessed answers, score) |
|---|---|---|---|---|---|
| 1 | What type of boats are docked in the rows? → *a boat* | 8 | What color are the boats in the rows? → *black* | 5 | ★ red, white, blue (0.78)<br>lake, ocean, river (0.62)<br>large boats, row boats, motorboats (0.54) |
| 2 | What color are the boats docked in the rows? → *azure* | 30 | What type of water body is hosting the boats docked in straight rows? → *a harbor* | 5 | ★ lake, river, ocean (0.53)<br>parallel, straight, aligned (0.02)<br>parallel, straight, aligned (0.02) |
| 3 | What type of water body are the boats docked in? → *azure* | 82 | What is the approximate number of boats in each row of the harbor? → *0* | 6 | ★ five, ten, several (0.05) |
| 4 | what is the other object in the image? → *a boat is docked in a boat moored in a boat* | 111 | What is the total number of rows of boats docked in the harbor? → *0* | 27 | yes, the rows are parallel, no boats present, rows are straight and empty (0.14)<br>zero boats docked (0.00)<br>★ zero rows, no boats, empty harbor (0.21)<br>zero rows, no boats, empty rows (0.14) |
| 5 | what is the other object in the image? → *a boat moored in a boat moored in a boat* | 32 | what is the other object in the image? → *a harbor with no boats docked in any rows* | 35 | ★ harbor structure, water surface, docks present (0.46)<br>zero, no boats, none (0.13)<br>no boats, zero boats, empty harbor (0.22) |
| 6 | What is the arrangement of the boats in the rows? → *azure* | 28 | What is the weather condition in the harbor scene? → *a harbor with no boats docked in any rows, featuring only the harbor structure and no* | 82 | ★ sunny, cloudy, rainy (0.59)<br>zero, no boats, no vessels (0.09) |
| 7 | What specific type of boats are docked in the rows? → *azure boats* | 41 | What is the weather condition in the harbor scene with no boats docked? → *a harbor with no boats docked* | 34 | empty, desolate, mysterious (0.21)<br>★ overcast, rainy, cloudy (0.34)<br>overcast, rainy, cloudy (0.26) |
| 8 | What type of boats are docked in the rows — wooden canoes or fishing boats? → *azure boats are docked in several rows* | 67 | What time of day is depicted in the harbor scene with no boats docked? → *a harbor with no boats docked in any rows* | 102 | ★ dawn, twilight, nighttime (0.20)<br>no boats in any row (0.00)<br>no boats docked (0.00) |
| 9 | What is the likely purpose or setting of the boats docked in straight rows? → *azure boats are docked in a marina* | 15 | What is the material of the dock where the boats are supposed to be docked? → *a harbor with no boats docked* | 35 | ★ wooden dock, concrete dock, steel dock (0.56) |
| 10 | What is the weather condition in the scene with the docked boats? → *azure boats in a marina* | 26 | What is the material of the dock where the boats are supposed to be docked? → *a harbor with no boats docked* | 101 | ★ wooden, concrete, metal (0.37) |

**PlugIR final search:** azure boats docked in several straight rows in a marina, with wooden canoes or fishing boats not present, all boats appearing to be of the same type and color, in a calm weather setting  
**Agent final search:** a harbor with no boats docked, featuring an empty dock and no rows of vessels, under unknown weather conditions, at an unspecified time of day, with no boats or visible vessels present

## 10. [000000255061.jpg](http://images.cocodataset.org/unlabeled2017/000000255061.jpg): PlugIR rank 498, agent rank 673

**Caption:** a woman swinging at a tennis ball on the court  
**Rank before any questions:** 92

| Round | PlugIR: question → answer | Rank | Agent: question → answer | Rank | Agent's options (guessed answers, score) |
|---|---|---|---|---|---|
| 1 | What color shirt is the woman wearing when she is swinging at the tennis ball? → *white* | 456 | What color is the woman's shirt? → *white* | 456 | ★ white, yellow, blue (1.03)<br>white (0.00)<br>white and blue (0.00) |
| 2 | What is the background setting of the tennis court where the woman is swinging? → *a woman in a white shirt is swinging at a tennis ball on a tennis* | 452 | What is the woman’s position in the swing? → *a woman in a white shirt is swinging a tennis racket at a* | 440 | ★ swinging at the ball, hitting the ball, in motion (0.09)<br>tennis racket, racquet in hand, holding racket (0.03)<br>none, racket, not visible (0.04)<br>swinging at the ball, hitting the ball, attacking the ball (0.05) |
| 3 | What is the exact activity of the woman besides swinging at the tennis ball? → *a woman in a white shirt is swinging at a tennis ball on a tennis* | 514 | What color are the woman’s shorts? → *white* | 768 | ★ white, no shorts, not specified (0.19) |
| 4 | What is the exact activity of the woman besides swinging at the tennis ball? → *a woman is swinging her arms at the tennis ball* | 336 | **Question: Is the woman hitting the tennis ball or just swinging the racket? → *a woman in white shirt and white shorts is swinging a tennis racket at* | 750 | tennis ball, racket, court (0.05)<br>swinging racket at ball (0.00)<br>★ hitting the ball, swinging the racket, both actions happening (0.13) |
| 5 | What color shorts is the woman wearing while swinging at the tennis ball? → *white* | 661 | **Question: Is the woman hitting the tennis ball or merely swinging the racket? → *a woman in a white shirt and white shorts is swinging a tennis racke* | 672 | tennis ball, racket, court (0.07)<br>hitting the ball, swinging the racket, both actions happening (0.13)<br>★ hitting the tennis ball, swinging the racket, both actions occur (0.13) |
| 6 | What type of footwear is the woman wearing while swinging at the tennis ball? → *a white shirt and white shorts* | 780 | what is the other object in the image? → *a tennis ball* | 725 | ★ tennis racket, tennis ball, white shorts (0.09)<br>swinging without contact, not making contact, just swinging racket (0.05)<br>swinging without hitting, just swinging racket, no contact made (0.04)<br>swinging without hitting, just swinging racket, no contact made (0.05) |
| 7 | What color shoes is the woman wearing while swinging at the tennis ball? → *white* | 662 | what is the other object in the image? → *a woman in a white shirt and white shorts is swinging a tennis racke* | 790 | ★ tennis racket, tennis ball, white shorts (0.09)<br>actively hitting the ball, swinging without contact, hitting the ball with racket (0.02) |
| 8 | What is the woman's posture or stance while swinging at the tennis ball? → *a woman in white shirt and white shorts swings her arms at a tennis ball on* | 541 | **Question: Is the woman on the court playing an active game of tennis or just practicing a swing? → *yes* | 685 | ★ playing an active game, just practicing a swing, mid-game action (0.04)<br>playing actively, taking a swing, in the middle of a game (0.02) |
| 9 | what is the other object in the image? → *a woman in white shirt and white shorts swings her arms at a tennis ball on* | 520 | **Question: Is the woman in the image on a red or white tennis court? → *no* | 607 | ★ white tennis court, red tennis court, not specified (0.06)<br>white, red, not specified (0.04) |
| 10 | what is the other object in the image? → *a woman in a white shirt and white shorts swings her arms at a tennis* | 498 | what is the other object in the image? → *a woman in a white shirt and white shorts is swinging a tennis racke* | 673 | ★ tennis racket, tennis ball, white shorts (0.03)<br>actively hitting the ball, swinging in preparation, about to hit the ball (0.01)<br>actively hitting the ball, swinging in preparation, about to hit the ball (0.01) |

**PlugIR final search:** a woman in a white shirt and white shorts swings her arms at a tennis ball on a tennis court, wearing white shoes, in a neutral stance, with no other objects visible in the scene  
**Agent final search:** A woman wearing a white shirt and white shorts is practicing a tennis swing by actively hitting a tennis ball with a racket on a white tennis court.
