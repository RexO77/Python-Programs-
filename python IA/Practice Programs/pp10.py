class shapes:
    def __init__(self):
        pass
class Triangle(shapes):
    def __init__(self,base,height):
        self.base = base
        self.height = height
    def area(self):
        return 0.5*self.base*self.height
class Circle(shapes):
    def __init__(self,rad):
        self.rad = rad
    def area(self):
        return 3.142*self.rad*self.rad
class Rectangle(shapes):
    def __init__(self,length,width):
        self.length = length
        self.width = width
    def area(self):
        return self.length*self.width
b= int(input("Enter Base :"))
h = int(input("Enter height:"))
triangle = Triangle(b,h)
print("Area of Triangle is : ",triangle.area())


