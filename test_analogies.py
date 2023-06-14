import matplotlib.pyplot as plt
import sklearn.decomposition
from gensim.models import fasttext, KeyedVectors

from cbow import CBOWModeler
import torch
from config import *
from utils import nearest_neighbour
from sklearn.manifold import TSNE

def main():
    draw_plot_from_explicit_accuracies()

    accuracy_list = []
    for i in range(7):
        print(f"Turn = {i}")

        path = "./en/cbow200/model" + str(i) + ".pth"
        checkpoint = torch.load(path, map_location=torch.device('cpu'))

        idx_to_word = checkpoint["idx_to_word"]
        word_to_idx = checkpoint["word_to_idx"]

        model = CBOWModeler(len(idx_to_word), 200)
        model.load_state_dict(checkpoint["cbow_state_dict"])
        embeds = model.embeddings.weight.data.cpu()

        accuracy = test_embeddings_on_analogies(idx_to_word, word_to_idx, embeds)
        accuracy_list.append(accuracy)

    print(accuracy_list)

    plt.plot([i for i in range(5)], accuracy_list)
    plt.ylabel("accuracy")
    plt.show()






def test_embeddings_on_analogies(idx_to_word:list[str], word_to_idx:dict[str:str], embeds:torch.Tensor, ban_list:list=[], path_analogies:str=""):
    def vec(word):
        return embeds[word_to_idx[word]]

    counter_all_examples = 0
    counter_correct_prediction = 0

    # retrieve analogies
    all_analogies_examples = get_analogies_examples(path_analogies)
    for analogy_type, analogies in all_analogies_examples.items():
        if analogy_type in ban_list:
            continue

        #try every valid analogy
        for index, analogy in enumerate(analogies):
            if all(item in idx_to_word for item in analogy):

                print(index, len(analogies))
                counter_all_examples+=1

                inp = vec(analogy[1]) - vec(analogy[0]) + vec(analogy[2])

                #get k most similar token...
                emb_ranking_top, euclidean_dis_top = nearest_neighbour(inp, embeds, 13)
                #...and remove tokens that come from the analogy
                emb_ranking_top = [idx_to_word[elt] for elt in emb_ranking_top if idx_to_word[elt] not in analogy[:-1]]

                #if expected result
                if analogy[3] in emb_ranking_top[:10]:
                    counter_correct_prediction+=1

    return counter_correct_prediction/counter_all_examples

def retrieve_tsne(embeds):

    model = TSNE(n_components = 2)
    embeds_2D = model.fit_transform(embeds)
    torch.save(obj=embeds_2D, f="./en/TSNE/2D_embeddings.pt")

    model = TSNE(n_components = 3)
    embeds_3D = model.fit_transform(embeds)
    torch.save(obj=embeds_3D, f="./en/TSNE/3D_embeddings.pt")


#sorry that's disgusting code
def draw_plot_from_explicit_accuracies():
    antonym_adjectives = [0.16996047430830039, 0.17193675889328064, 0.16996047430830039, 0.16403162055335968,
                                     0.13438735177865613]  # 601
    gram6_nationality_adjective = [0.013461538461538462, 0.009615384615384616, 0.005128205128205128,
                                              0.009615384615384616, 0.007692307692307693]  # 1640
    currency = [0.02619047619047619, 0.02142857142857143, 0.011904761904761904, 0.02619047619047619,
                           0.04047619047619048]  # 813
    gram1_adjective_to_adverb = [0.06435643564356436, 0.05693069306930693, 0.06930693069306931,
                                            0.06930693069306931, 0.03094059405940594]  # 927
    gram2_opposite = [0.03306878306878307, 0.02513227513227513, 0.03042328042328042, 0.031746031746031744,
                                 0.0873015873015873]  # 757
    # capital_world = [] # 12211
    capital_common_countries = [0.013157894736842105, 0.015789473684210527, 0.018421052631578946,
                                           0.010526315789473684, 0.0]  # 381
    family = [0.3380952380952381, 0.30952380952380953, 0.3380952380952381, 0.3380952380952381,
                         0.3952380952380952]  # 273
    city_in_state = [0.0014240506329113924, 0.00047468354430379745, 0.0007911392405063291,
                                0.0009493670886075949, 0.0006329113924050633]  # 7833

    for acc, name in zip([antonym_adjectives, gram6_nationality_adjective, currency, gram1_adjective_to_adverb, gram2_opposite, capital_common_countries, family, city_in_state], ["antonym_adjectives", "gram6_nationality_adjective", "currency", "gram1_adjective_to_adverb", "gram2_opposite", "capital_common_countries", "family", "city_in_state"]):
        plt.plot([i for i in range(5)], acc, label=name)
    plt.legend(title="Type d'analogie", loc="center", bbox_to_anchor=(1.2, 0.5))
    plt.title("Score sur les analogies françaises selon chaque itération\n(par type d'analogie)")
    plt.xlabel("itérations")
    plt.ylabel("score")
    plt.ylim(0, 1)
    plt.show()

    gram3_comparative = [0.9159159159159159, 0.9512012012012012, 0.9722222222222222, 0.9669669669669669,
                         0.978978978978979, 0.9722222222222222, 0.9752252252252253]
    gram6_nationality_adjective = [0.8599562363238512, 0.9328956965718453, 0.9518599562363238, 0.9489423778264041,
                                   0.9518599562363238, 0.9467541940189642, 0.9511305616338439]
    family = [0.8, 0.7952380952380952, 0.7880952380952381, 0.7833333333333333, 0.780952380952381, 0.7833333333333333,
              0.7785714285714286]
    gram9_plural_verbs = [0.8017241379310345, 0.833743842364532, 0.8583743842364532, 0.8620689655172413,
                          0.8620689655172413, 0.8583743842364532, 0.8645320197044335]
    gram8_plural = [0.7210338680926917, 0.8440285204991087, 0.8609625668449198, 0.8627450980392157, 0.8609625668449198,
                    0.8672014260249554, 0.8689839572192514]
    gram7_past_tense = [0.7192307692307692, 0.8237179487179487, 0.8391025641025641, 0.8762820512820513,
                        0.8826923076923077, 0.8846153846153846, 0.8935897435897436]
    gram5_present_participle = [0.6743951612903226, 0.7268145161290323, 0.7772177419354839, 0.782258064516129,
                                0.7903225806451613, 0.7893145161290323, 0.7893145161290323]
    gram4_superlative = [0.6491935483870968, 0.7106854838709677, 0.7449596774193549, 0.7479838709677419,
                         0.7328629032258065, 0.7419354838709677, 0.7469758064516129]
    capital_common_countries = [0.6363636363636364, 0.7035573122529645, 0.7628458498023716, 0.7865612648221344,
                                0.8023715415019763, 0.8221343873517787, 0.8280632411067194]
    capital_world = [0.3686779059449867, 0.4951197870452529, 0.546583850931677, 0.5368234250221828, 0.5252883762200532,
                     0.5399290150842946, 0.5359361135758651]
    city_in_state = [0.35673289183222956, 0.2860927152317881, 0.23487858719646798, 0.2097130242825607,
                     0.19911699779249448, 0.20485651214128037, 0.20838852097130242]
    gram1_adjective_to_adverb = [0.2711693548387097, 0.3397177419354839, 0.3659274193548387, 0.39314516129032256,
                                 0.38810483870967744, 0.3860887096774194, 0.3870967741935484]
    gram2_opposite = [0.25213675213675213, 0.301994301994302, 0.31196581196581197, 0.33475783475783477,
                      0.3176638176638177, 0.3148148148148148, 0.3076923076923077]
    currency = [0.019736842105263157, 0.05263157894736842, 0.06578947368421052, 0.05263157894736842,
                0.05921052631578947, 0.05921052631578947, 0.06578947368421052]

    for acc, name in zip    ([gram3_comparative, gram6_nationality_adjective, family, gram9_plural_verbs, gram8_plural, gram7_past_tense, gram5_present_participle, gram4_superlative, capital_common_countries, capital_world, city_in_state, gram1_adjective_to_adverb, gram2_opposite, currency],["gram3_comparative", "gram6_nationality_adjective", "family", "gram9_plural_verbs", "gram8_plural", "gram7_past_tense", "gram5_present_participle", "gram4_superlative", "capital_common_countries", "capital_world", "city_in_state", "gram1_adjective_to_adverb", "gram2_opposite", "currency"]):
        plt.plot([i for i in range(7)], acc, label=name)
    plt.legend(title="Types of analogies", loc="center", bbox_to_anchor=(1.2, 0.5))
    plt.legend(title="Type d'analogie", loc="center", bbox_to_anchor=(1.2, 0.5))
    plt.title("Score sur les analogies anglaises selon chaque itération\n(par type d'analogie)")
    plt.xlabel("itérations")
    plt.ylabel("score")
    plt.ylim(0, 1)
    plt.show()
def get_analogies_examples(path:str) -> dict[str:list[str]]:

    with open(path, "r", encoding="utf-8") as file:
        #load whole file
        text = file.read().strip()
        #split into each class of analogies
        analogies = text.lower().split(":")[1:]
        #split each example
        analogies = {analogy.split("\n")[0].strip(): analogy.split("\n")[1:] for analogy in analogies}
        #split each token within each example
        analogies = {analogy_type:[example.split(" ") for example in analogies.get(analogy_type)] for analogy_type in analogies.keys()}
    return analogies

def plot_fasttext():

    words = ["describes","icelandic","eagle","discovering","unlikely","generated","brighter","discover","brothers","stepmother","macedonian","pittsburgh","poland","sons","speaks","searches","kuna","buildings","quietly","stronger","forint","capital-world","somalia","kansas","cyprus","australian","malaysia","men","elephant","clouds","unaware","largest","georgia","reading","indian","smallest","decrease","chilean","bishkek","minnesota","garland","estimates","egypt","austria","finland","striking","uncompetitive","jakarta","mangoes","furiously","hand","mice","girl","predicting","increasing","tulsa","france","safer","sat","said","slows","screamed","germany","russia","heavy","reading","precisely","afghanistan","belorussian","youngest","naira","wichita","lebanon","family","modesto","cheaper","libya","amazingly","shrinking","rupee","indianapolis","slowly","tennessee","gram8-plural","riel","tennessee","somalia","louisiana","laos","chicago","safe","brazilian","melons","italian","lima","worst","kingston","smarter","lithuania","paris","oregon","clear","predicting","walks","uruguay","grandma","belgium","argentinean","iran","said","oldest","quickly","cows","works","bahrain","pakistan","laredo","senegal","quick","tokyo","niger","oldest","mexican","austrian","vanished","uzbekistan","lucky","tunis","morocco","algeria","bride","grandfather","wrote","low","debugging","sisters","nebraska","gram1-adjective-to-adverb","paying","laos","kazakhstan","tighter","georgia","swift","infrequently","hungary","bhutan","inefficient","gram2-opposite","lilongwe","denar","decreasing","dollar","iraq","dancing","japanese","girl","read","cars","denmark","slower","texas","dad","maltese","queen","egyptian","says","lev","mostly","german","minsk","leu","unreasonable","coolest","her","older","swiss","conakry","shrank","dream","dreams","smarter","serbia","bad","shuffling","dram","nuuk","took","thinks","belorussian","tuvalu","hands","grandma","immediate","undecided","malawi","rats","colder","sure","namibia","gram3-comparative","dollar","alaska","luckily","rapidly","arizona","efficiently","easier","tough","weirdest","inventing","cows","implemented","tallest","impossibly","cambodian","thought","knowing","honest","australian","bakersfield","anaheim","illinois","won","freely","calmly","she","denar","listened","aware","easy","greatest","liberia","chinese","ran","longer","tasty","pineapple","tunisia","darkest","talks","baghdad","described","completely","portuguese","easier","irrational","zloty","stockton","luckiest","ottawa","sold","warmest","norway","unsure","swam","latvia","flew","speaks","angola","onion","dollars","sweetest","tajikistan","pennsylvania","jumping","apparent","bahamas","bucharest","botswana","madrid","looked","biggest","stepdaughter","uncle","bamako","ethical","honolulu","massachusetts","describing","fremont","birds","florida","singing","easiest","baltimore","vaduz","greenland","malawi","louisville","mother","tallinn","cat","ghana","unconvincing","enhancing","euro","largest","forint","fingers","hawaii","greatest","washington","knew","liberia","woman","jumped","washington","finger","consistent","belmopan","daughters","zimbabwe","hid","qatar","predict","mogadishu","philippines","litas","played","better","rwanda","guyana","ghana","norwegian","bananas","dogs","cooler","ecuador","elephants","german","implement","monkeys","computers","better","coding","bananas","kampala","bujumbura","swimming","nigeria","hiding","havana","unconvincing","flying","precisely","distasteful","eyes","austria","jumping","brazilian","moscow","illinois","works","old","sits","spent","bhutan","zambia","she","venezuela","rat","gram5-present-participle","saw","running","strongest","israeli","azerbaijan","roseau","convincing","predicts","cleveland","niamey","lithuania","amarillo","hungary","nicaragua","hit","new","suddenly","weakest","belarus","spanish","feeding","nigeria","michigan","fresno","enhances","shuffles","warmer","poland","tbilisi","unimpressive","belarus","younger","dublin","slovakian","managua","run","rabat","nashville","colors","omaha","smallest","lisbon","lowest","fortunately","bulgaria","bigger","dog","pears","muscat","peso","safely","smartest","impossibly","illogical","brighter","pear","writes","slow","good","policewoman","liechtenstein","mom","colombian","lats","hialeah","smart","goat","coldest","grandson","enhancing","shuffling","greek","cyprus","weak","akron","azerbaijan","decreases","plays","unknown","decreased","buildings","walked","amazing","cheerfully","informed","highest","screams","unknown","georgetown","typically","greece","implemented","taiwan","undecided","child","weaker","sweetest","france","impossible","inventing","cheerfully","enhanced","italy","thought","horse","bulgaria","snakes","provide","moldovan","sharper","stepdaughter","describe","swiftly","bangladesh","unacceptable","enhanced","memphis","naira","cheap","cheaper","luanda","nassau","searches","kenya","fastest","impossible","dutch","serbia","harder","known","won","dollars","worcester","mango","korean","stronger","niece","stepmother","senegal","nevada","sharpest","deeper","switzerland","walking","iran","cairo","damascus","rapid","irresponsible","work","amazingly","burundi","walk","enhance","color","sees","venezuela","larger","jordan","thailand","pleasant","detroit","granddaughter","denmark","heavier","huntsville","anchorage","father","sings","irresponsible","peruvian","antananarivo","bigger","darkest","canberra","simplest","tunisia","car","lebanon","arizona","policeman","implementing","moved","certain","unlikely","building","saying","syria","flying","faster","king","typical","generated","mali","invent","smaller","eats","decreased","moldova","pakistan","efficient","rational","jacksonville","afghanistan","skopje","tacoma","best","listens","horses","aunt","fortunate","malta","ohio","england","macedonian","generates","pigs","california","beijing","colors","reno","macedonia","ankara","thai","egypt","chile","thai","cheerful","polish","lusaka","computer","niece","tallest","indonesia","sofia","vilnius","obviously","faster","jamaica","quicker","men","yen","unproductive","unpleasant","most","implements","mexican","dhaka","widest","mostly","india","leu","saying","coolest","decreases","reluctantly","tuvalu","japan","bottle","wisconsin","sitting","cats","scottsdale","ukraine","immediately","malta","kathmandu","slower","lats","freely","helsinki","estimate","gabon","clouds","milwaukee","oman","sudan","bright","increases","peso","slows","bottles","donkeys","uncompetitive","harder","belize","daughter","simpler","grandpa","lions","taller","austin","city-in-state","larger","occasionally","moved","suddenly","ran","happily","danish","china","irish","increases","monrovia","croatia","thailand","stockholm","sharpest","sits","unimpressive","birds","jumped","see","sold","vanishing","armenia","namibia","lexington","inconsistent","usually","alabama","fontana","chisinau","estonia","belgium","goes","decided","tastiest","rats","screamed","gram7-past-tense","policewoman","granddaughter","slovakia","libya","dinar","thinking","israel","litas","hryvnia","wife","chile","dong","hottest","polish","armenia","obvious","shrank","longest","fell","gram9-plural-verbs","longer","slowest","weaker","liechtenstein","chandler","argentina","uninformative","eye","sing","dutch","english","sang","code","turkmenistan","taiwan","taller","warmer","nairobi","hanoi","chinese","boston","japan","tasteful","listened","sudden","mozambique","suriname","doha","fast","finland","mangoes","obviously","man","irish","albania","unethical","oxnard","fastest","nepal","pennsylvania","budapest","toledo","strong","wider","ukrainian","professionally","tall","stepsister","euro","mesa","seriously","unfortunate","spent","honduras","vientiane","morocco","bride","distasteful","write","wife","looking","monkeys","korea","gabon","sister","coldest","lower","shortest","rwanda","bahrain","vanishes","baht","unclear","productive","dinar","walked","best","talk","albanian","hotter","think","yen","finds","hit","australia","indonesia","usual","louder","santiago","greek","melons","describes","massachusetts","newer","english","weirdest","dominica","bangladesh","gaborone","smaller","paid","norwegian","russian","jordan","china","tajikistan","florida","maputo","describing","albania","computers","sisters","quicker","strangest","iraq","stepson","slowest","slowly","ukrainian","coding","brother","convenient","uninformed","quickest","bird","turkey","possibly","easiest","plano","deeper","weird","increase","screaming","safer","guyana","slovakia","precise","calmly","pigs","uninformative","inefficient","vanish","croatia","walking","higher","hitting","long","big","krone","unfortunately","uzbekistan","maryland","orlando","played","strange","glendale","lower","higher","listens","hard","sweden","indiana","tallahassee","highest","cuba","cats","turkmenistan","complete","brightest","arlington","small","sacramento","athens","singing","roads","nicosia","ashgabat","greenland","colorado","italy","apia","currency","fell","austrian","kentucky","debug","danced","implements","increased","colder","predicts","kiev","shortest","dark","bottles","portugal","logical","fortunately","pears","caracas","sat","scream","baht","dogs","children","his","strongest","switzerland","bulgarian","increasing","zagreb","informative","cuba","kazakhstan","eagles","melon","furious","taking","stepsister","free","macedonia","onions","bratislava","generating","kuna","hryvnia","uganda","copenhagen","swedish","yerevan","thimphu","slowed","lion","discovering","podgorica","quietly","nepal","simpler","eat","turkey","dishonest","slovenia","dallas","running","fed","sings","unsure","swims","swim","princess","goats","walks","tougher","algeria","tougher","russian","samoa","uncertain","cambodian","capital-common-countries","shorter","rupee","brightest","moldova","seeing","quickly","oklahoma","cloud","machines","banjul","bahamas","slept","decreasing","went","irving","zambia","ruble","occasionally","slept","machine","children","texas","knew","ljubljana","zimbabwe","ukraine","smartest","immediately","krona","illogical","maryland","accra","windhoek","wrote","gambia","writing","listen","princess","unproductive","greece","greater","provides","canada","unfortunate","moldovan","possibly","russia","snake","tirana","brazil","madison","loud","inconsistent","mom","play","eagles","oslo","dong","kabul","belize","suriname","high","ecuador","jamaica","looking","women","hands","kansas","unfortunately","swedish","going","tighter","banana","groom","vienna","professionally","playing","germany","daughter","fiji","serious","inconvenient","honduras","monkey","oakland","fly","kigali","spanish","baku","daughters","dancing","provides","greater","dram","tashkent","tripoli","mother","danish","europe","heavier","woman","selling","older","fiji","hid","shorter","large","phoenix","donkey","unreasonable","pig","french","alaska","oregon","talks","indiana","stepfather","latvia","quito","uninformed","croatian","paramaribo","worse","warsaw","ruble","chilean","donkeys","her","unaware","stepbrother","montenegro","slowing","find","kwanza","portland","reluctantly","croatian","real","quickest","safely","abuja","kwanza","real","son","cooler","swiss","guinea","says","rarely","dakar","peru","pineapples","falling","hot","funafuti","korean","sweet","likely","sees","mauritania","screams","lowest","hottest","aunt","philippines","saw","krona","say","sister","beirut","dance","irvine","usa","lions","tehran","implementing","romania","nebraska","listening","danced","bern","grandmother","unpleasant","vietnam","belgrade","luckily","mexico","dreams","quiet","spain","spokane","asmara","ireland","lubbock","italian","burundi","suva","sleeping","newer","ireland","screaming","sitting","french","rapidly","struck","eritrea","shreveport","read","kyrgyzstan","dishonest","sang","worst","going","completely","flew","tegucigalpa","boy","tampa","egyptian","husband","infrequently","spending","acceptable","seeing","louder","spain","worse","move","riel","took","sudan","plays","cool","usually","deep","angola","israeli","jump","generating","icelandic","longest","snakes","tucson","enhances","eritrea","cincinnati","writing","writes","inconvenient","furiously","amman","paid","khartoum","mozambique","horses","nouakchott","playing","look","warmest","comfortable","rarely","listening","england","kenya","slovakian","calm","mouse","miami","slovenia","houston","gram4-superlative","estimates","colorado","samoa","peru","kentucky","moving","prince","goats","indian","cow","uncomfortable","netherlands","portugal","harare","unclear","fingers","efficiently","algiers","impressive","montevideo","professional","kyrgyzstan","wide","montenegro","goes","occasional","libreville","portuguese","zloty","peruvian","competitive","onions","roads","thinking","bangkok","vietnam","happily","mauritania","wisconsin","minnesota","madagascar","estonia","sharp","valletta","unacceptable","road","alabama","sweden","responsible","astana","guinea","gram6-nationality-adjective","generates","vanishing","apparently","henderson","youngest","uruguay","nicaragua","hotter","swiftly","philadelphia","dominica","maltese","weakest","albanian","wider","typically","colombia","search","krone","nephew","slowed","islamabad","rial","eats","ringgit","elephants","swam","shuffles","london","biggest","apparently","irrational","pineapples","syria","tight","generate","shuffle","argentinean","widest","speak","younger","california","australia","riga","eyes","cold","short","rare","young","grandmother","queen","manama","thinks","simplest","madagascar","moving","reluctant","mice","struck","minneapolis","slowing","predicted","mali","atlanta","denver","uncomfortable","bulgarian","finds","women","dushanbe","rome","canada","cars","increased","vanished","reasonable","debugging","louisiana","romania","hawaii","great","cambodia","michigan","fed","possible","sit","vanishes","ohio","botswana","unethical","warm","machines","seattle","uncertain","japanese","simple","niger","iceland","nevada","swimming","brussels","described","went","lev","colombian","ringgit","strangest","infrequent","seriously","happy","uganda","swims","luckiest","gambia","taipei","manila","qatar","looked","sharper","oman","norway","rial","berlin","oklahoma","tastiest","predicted"]
    # retrieve fasttext vectors
    # run only once !
    model_fasttext = fasttext.load_facebook_model("./en/fastext/cc.en.300.bin")
    model_kv = model_fasttext.wv
    new_vectors = model_kv.vectors_for_all({word:1 for word in words if word in model_kv.key_to_index})
    new_vectors.save('./en/fastext/fastext_en_allanalogies_vectors.kv')

    # run everytime !
    # load fasttext vectors
    # loaded_vectors = KeyedVectors.load('./en/fastext/fastext_fr_vectors.kv')
    # fasttext_vectors = [loaded_vectors.get_vector(lemma) for lemma in loaded_vectors.key_to_index.keys()]
    #
    # test_embeddings_on_analogies(loaded_vectors.index_to_key(), loaded_vectors.key_to_index(), fasttext_vectors)

def plot_analogy(idx_to_word:list[str], word_to_idx:dict[str:str], embeds:torch.Tensor) -> None:
    all_analogies_examples = get_analogies_examples("questions-words.txt")

    model = TSNE(n_components = 3)
    embeds = model.fit_transform(embeds)

    A, B, C = embeds[word_to_idx["man"]], embeds[word_to_idx["king"]], embeds[word_to_idx["woman"]]
    tautology = B - A

    x, y, z = [elt[0] for elt in [A, B, C, tautology]], [elt[1] for elt in [A, B, C, tautology]], [elt[2] for elt in [A, B, C, tautology]]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, c=['y', 'y', 'r', 'g'])
    plt.show()

if __name__ == "__main__":
    main()
