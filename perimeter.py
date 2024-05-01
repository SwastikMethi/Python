#Swastik Methi
#E22CSEU1468
import math
class shape:
    print("Choose the shapes below:")
    def __init__(self):
        self.area = 0
        self.side = 0
        self.radius = 0
        self.length = 0
        self.width = 0
        self.height = 0
        self.perimeter = 0
        self.choice = 0
        self.pi = math.pi
        self.menu()
    def menu(self):
        print("1. Circle")
        print("2. Rectangle")
        print("3. Square")
        print("4. Hexagon")
        print("5. Exit")
        self.choice = int(input("Enter your choice: "))
        if self.choice == 1:
            self.circle()
        elif self.choice == 2:
            self.rectangle()
        elif self.choice == 3:
            self.square()
        elif self.choice == 4:
            self.hexagon()
        elif self.choice == 5:
            exit()
        else:
            print("Invalid choice")
            self.menu()
    def circle(self):
        self.radius = float(input("Enter the radius of the circle: "))
        self.area = self.pi * self.radius * self.radius
        self.perimeter = 2 * self.pi * self.radius
        print("Area of the circle is: ", self.area)
        print("Perimeter of the circle is: ", self.perimeter)
        self.menu()

    def rectangle(self):
        self.length = float(input("Enter the length of the rectangle: "))
        self.width = float(input("Enter the width of th width: "))
        self.area = self.length * self.width
        print("Area of the rectangle is: ", self.area)
        print("Perimeter of the rectangle is: ", 2 * (self.length + self.width))
        self.menu()

    def square(self):
        self.side = float(input("Enter the side of the Square: ")) 
        self.area = self.side * self.side
        self.perimeter = 4* self.side
        print("Area of the square is: ", self.area)
        print("Perimeter of the square is: ",self.perimeter)
        self.menu()

    def hexagon(self):
        self.side = float(input("Enter the side of the hexagon: "))
        self.area = ((3 * 1.732 * self.side * self.side) / 2)
        self.perimeter = 6 * self.side
        print("Area of the hexagon is: ", self.area)
        print("Perimeter of the hexagon is: ",self.perimeter)
        self.menu()

s = shape()

    

