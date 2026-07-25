#类
class stu():
    def __init__(self, name, age):
        self.__name = name
        self.__age = age
    def show(self):
        print(f"hello {self.__name}, you are {self.__age} now")
student = stu("yucheng", 19)
student.show()
