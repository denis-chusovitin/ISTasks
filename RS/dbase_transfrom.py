def get_rating(x):
    if x == 1:
        return 1
    elif x <= 5:
        return 2
    elif x <= 10:
        return 3
    elif x <= 20:
        return 4
    else:
        return 5

def get_value(dict, key):
    if key not in dict:
        dict[key] = len(dict)
    return dict[key]

users_dictionary = {}
tracks_dictionary = {}

with open('train_triplets.txt', 'r') as r_file, open('triplets.txt', 'w') as w_file:
    for line in r_file:
        triplet = line[:-1].split('\t')

        user_id = get_value(users_dictionary, triplet[0])
        track_id = get_value(tracks_dictionary, triplet[1])
        r = get_rating(int(triplet[2]))

        w_file.write('{!s} {!s} {!s}\n'.format(user_id, track_id, r))
