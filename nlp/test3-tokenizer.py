from requests import get
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def get_file():
    BIG_FILE_URL = 'https://raw.githubusercontent.com/dscape/spell/master/test/resources/big.txt'
    with open('files/big.txt', 'wb') as big_f:
        response = get(BIG_FILE_URL, )

        if response.status_code == 200:
            big_f.write(response.content)
        else:
            print("Unable to get the file: {}".format(response.reason))


if __name__ == '__main__':
    # get_file()
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = Sequence([
        NFKC(),
        Lowercase()
    ])
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    # 设置词典
    trainer = BpeTrainer(vocab_size=25000, show_progress=True, initial_alphabet=ByteLevel.alphabet())
    tokenizer.train(trainer, ["files/big.txt"])

    print("Trained vocab size:{}".format(tokenizer.get_vocab_size()))
    # tokenizer.model.save('.')    # 保存模型

    # 加载模型
    tokenizer.model = BPE('files/vocab.json', 'files/merges.txt')
    encoding = tokenizer.encode("This is a simple input to be tokenized")

    print("Encoded string: {}".format(encoding.tokens))

    decoded = tokenizer.decode(encoding.ids)
    print("Decoded string: {}".format(decoded))
