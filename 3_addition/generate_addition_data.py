# Generate a dataset of addition problems
import random
import os

def generate_addition_data(n):
    digits_seen = set()
    data = []
    for _ in range(n*2):
        a = random.randint(1000, 9999)
        b = random.randint(1000, 9999)
        c = a + b
        if (a, b) in digits_seen or (b, a) in digits_seen:
            continue
        digits_seen.add((a, b))
        data.append((a, b, c))
        if len(data) == n:
            break
    # Format as string
    data = ['%d+%d=%d' % (a, b, c) for a, b, c in data]
    return data

if __name__ == '__main__':
    random.seed(24)
    data = generate_addition_data(5000000)
    d_train = data[:-10000]
    d_test = data[-10000:]

    # Save in a data directory
    os.makedirs('data', exist_ok=True)
    with open('data/addition_train.txt', 'w') as f:
        f.write('\n'.join(d_train))

    with open('data/addition_test.txt', 'w') as f:
        f.write('\n'.join(d_test))

    print('Train:', len(d_train), 'Test:', len(d_test))
