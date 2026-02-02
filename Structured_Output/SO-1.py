from typing import TypedDict

class Person(TypedDict):
    name:str
    age: int

person1 :Person = {'name':'Hriday','age':20}

print(person1)
