import PySimpleGUI as sg
layout=[
    [sg.Text("Press to Register")]
    [sg.Button("OK")]
   ]

window=sg.Window("Demo",layout)
while True:
    event, values = window.read()
    if event == "OK" or event == sg.WIN_CLOSED:
        break

window.close()
# sg.Window(title="Play'O'SOL",layout=[[]], margins=(250,250)).read()
# sg.button("Play")