from tkinter import*
root = Tk()

mylabel1 =Label(root,text="Hello world")
mylabel2 =Label(root,text="Swastik")

mylabel1.grid(row=5,column=8)
mylabel2.grid(row=4,column=8)

root.mainloop()