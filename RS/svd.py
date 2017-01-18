import math

users_amount = 1019317

gamma = 0.001
lambda1 = 0.005
lambda2 = 7
k = 100

total = 0

def vec_prod(a, b):
    return sum(map(lambda x, y: x * y, a, b))

def update_predictor(b):
    return b + gamma * (eui - lambda1 * b)

def read_data():
    user_id = 0
    global tracks_amount
    global total

    with open('triplets.txt', 'r') as file:
        for line in file:
            triplet = map(lambda x: int(x), line[:-1].split(' '))

            total += 1

            if triplet[0] > users_amount:
                break

            if tracks_amount < triplet[1]:
                tracks_amount = triplet[1]

            if (user_id != triplet[0]):
                data.append(([], []))
                user_id = triplet[0]

            data[triplet[0]][0].append(triplet[1])
            data[triplet[0]][1].append(triplet[2])

def predict(u, i):
    return mu + bi[i] + bu[u] + vec_prod(q[i], p[u])

def count_rmse():
    ratings = [data[u][1][i] for u in range(train_amount, users_amount) for i in range(len(data[u]))]

    return math.sqrt(sum(map(lambda r, r_p: (r - r_p)**2, ratings, predictions)) / len(ratings))

data = [([], [])]
tracks_amount = -1
train_amount = int(0.8 * users_amount)

print "Reading data.."
read_data()
print "Done."

print "Users: {!s}, Tracks: {!s}, Users-Tracks: {!s}, Train set: {!s}".format(
    users_amount, tracks_amount, total, train_amount)

mu = 0
bu = [0] * users_amount
bi = [0] * tracks_amount
p = [[0.1] * k] * users_amount
q = [[0.05 * i for i in range(k)]] * tracks_amount

rmse = 1
rmse_old = 0

iter = 0

threshold = 0.1

print "Training..."

while abs(rmse - rmse_old) > 0.0001:
    rmse_old = rmse
    rmse = 0

    for u in xrange(train_amount):
        user_songs_amount = len(data[u])

        for j in xrange(user_songs_amount):
            i = data[u][0][j]

            eui = data[u][1][j] - (mu + bu[u] + bi[i] + vec_prod(q[i], p[u]))

            rmse += eui * eui / total

            mu += gamma * eui
            bu[u] = update_predictor(bu[u])
            bi[i] = update_predictor(bi[i])

            for f in xrange(k):
                p[u][f] += gamma * (eui * q[i][f] - lambda2 * p[u][f])
                q[i][f] += gamma * (eui * p[u][f] - lambda2 * q[i][f])

    iter += 1
    rmse = math.sqrt(rmse)

    print "Iteration: {!s}, RMSE: {!s}".format(iter, rmse)

    if rmse > rmse_old - threshold:
        gamma = gamma * 0.66
print "Done."

print "Testing.."
predictions = [predict(u, i) for u in range(train_amount, users_amount) for i in range(len(data[u]))]
print "Done."

print "RMSE: ", count_rmse()
