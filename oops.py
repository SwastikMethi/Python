# Swastik Methi
#E22CSEU1468
from datetime import datetime

class User:
    def _init_(self, name, email):
        self.name = name
        self.email = email

class Seller:
    def _init_(self, name, products):
        self.name = name
        self.products = products

class Product:
    def _init_(self, name, price):
        self.name = name
        self.price = price

class Order:
    def _init_(self, user, seller):
        self.user = user
        self.seller = seller
        self.products_ordered = []
        self.total_price = 0
        self.date_time = datetime.now()

    def add_product(self, product):
        self.products_ordered.append(product)
        self.total_price += product.price

# Creating instances of products
product1 = Product("Product 1", 10)
product2 = Product("Product 2", 20)
product3 = Product("Product 3", 30)

# Creating instances of sellers with their product lists
seller1 = Seller("Seller 1", [product1, product2])
seller2 = Seller("Seller 2", [product2, product3])

# Creating user
user_name = input("Enter your name: ")
user_email = input("Enter your email: ")
user = User(user_name, user_email)

while True:
    print("Available sellers:")
    print("1. Seller 1")
    print("2. Seller 2")
    seller_choice = input("Select a seller (1/2): ")

    if seller_choice == '1':
        selected_seller = seller1
    elif seller_choice == '2':
        selected_seller = seller2
    else:
        print("Invalid choice.")
        continue

    print("Available products:")
    for index, product in enumerate(selected_seller.products, start=1):
        print(f"{index}. {product.name} - ${product.price}")

    product_choice = int(input("Select a product: ")) - 1
    selected_product = selected_seller.products[product_choice]

    another_product = input("Do you want to add another product? (yes/no): ")
    if another_product.lower() != 'yes':
        break

order = Order(user, selected_seller)
order.add_product(selected_product)

print("\nOrder details:")
print(f"User: {order.user.name} ({order.user.email})")
print(f"Seller: {order.seller.name}")
print("Products ordered:")
for product in order.products_ordered:
    print(f"- {product.name} (${product.price})")
print(f"Total Price: ${order.total_price}")
print(f"Date and Time: {order.date_time}")