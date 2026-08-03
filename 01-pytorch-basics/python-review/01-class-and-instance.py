"""Python 类、实例属性和方法的最小示例。"""


class Student:
    def __init__(self, name: str, age: int) -> None:
        self._name = name
        self._age = age

    def introduce(self) -> None:
        print(f"Hello {self._name}, you are {self._age} now.")


student = Student("Yucheng", 19)
student.introduce()
