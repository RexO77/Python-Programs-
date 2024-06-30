class Shape:
    def area(self):
        pass
class  Triangle(Shape):
    def __init__ (self, base, height): 
        self.base = base
        self.height = height 
    def area(self):
        return 0.5 * self.base * self.height 
class Circle(Shape):
    def __init__ (self, radius): 
        self.radius = radius
    def area(self):
        return 3.14 * self.radius * self.radius 
class Rectangle(Shape):
    def __init__(self, length, width): 
        self.length = length
        self.width = width 
    def area(self):
        return self.length * self.width 
print("--------------------------------------------------")
b=int(input("Enter the value of base :")) 
h=int(input("Enter the value of height :"))
triangle = Triangle(b,h)
print("Area of the Triangle :", triangle.area())
print("--------------------------------------------------")
r=int(input("Enter the value of radius :"))
circle = Circle(r)
print ("Area of the Circle :", circle.area())
print("--------------------------------------------------")
l=int(input("Enter the value of Length :"))
w=int(input("Enter the value of width :")) 
rectangle = Rectangle(l, w)
print ("Area of the Rectangle :", rectangle.area())
print("--------------------------------------------------")