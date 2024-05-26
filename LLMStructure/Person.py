import random
from name import Name


class Person:

    def __init__(self, key: str, name: str):
        self.primeKey = key
        '''=====Bioinfo Table====='''
        self.age = random.randint(10, 70)
        self.gender = random.choice(['male', 'female'])
        self.name = name
        self.height = random.randint(120, 220)
        self.weight = random.randint(70, 180)
        self.color = random.choice(Person.color())

        '''=====Work Table====='''
        self.status = random.choice(["retired", "employed", "unemployed"])
        self.salary = random.randint(1000, 1000_000)
        self.company = random.choice(["NVIDIA", "Apple", "Meta", "Google", "Twitter", "Microsoft", "OPENAI"])
        self.location = random.choice(Person.cities())

    def __bio(self):
        return f'{self.gender}\t{self.age}\t{self.name}\t{self.height}\t{self.weight}\t{self.color}'

    def __wrk(self):
        return f'{self.status}\t{self.salary}\t{self.company}\t{self.location}'

    def bio_to_string(self):
        return f'{self.primeKey}\t{self.__bio()}'

    def work_to_string(self):
        return f'{self.primeKey}\t{self.__wrk()}'

    def to_string(self):
        return f'{self.primeKey}\t{self.__bio()}\t{self.__wrk()}'

    @classmethod
    def gen_random(cls, simple=True):
        key_gen = Name()
        name_gen = Name(init=random.randint(0,25))
        size = random.randint(4, 10) if simple else random.randint(10, 15)
        data = [Person(key_gen.poll(), name_gen.poll()) for _ in range(size)]
        return data, list(key_gen.hist)

    @classmethod
    def bio_table_head(cls):
        return f'primeKey\tgender\tage\tname\theight\tweight\tcolor'

    @classmethod
    def work_table_head(cls):
        return f'primeKey\tstatus\tsalary\tcompany\tlocation'

    @classmethod
    def fields(cls):
        return ['age', 'gender', 'name', 'height', 'weight', 'color', 'status', 'salary', 'company', 'location']

    @classmethod
    def color(cls):
        return ['white', 'black', 'brown', 'olive', 'swarthy', 'mulatto']

    @classmethod
    def cities(cls):
        return ["CA", "WA", "TX", "NY", "GA", "HI", "IL"]
