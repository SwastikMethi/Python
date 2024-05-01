import pyautogui as pg;
import time

#\\code to auto write something//#
 
pg.countdown(10)

for i in range(1):
    text = ("""Enter your text here.""")
    pg.write(text, interval=0.01)
    pg.press("enter")


#\\code to close an application//#

# 01-----------------------------------------

# pg.countdown(5)
# pg.hotkey("alt","f4")

# 02-----------------------------------------

# pg.moveTo(2831,34)
# pg.click()
# pg.moveTo(1101,1753)
# pg.click()
# pg.countdown(2)
# text = ("""Chrome""")
# pg.write(text, interval=0.1)
# pg.countdown(2)
# pg.press('enter')

#\\Download an image from google//#
# pg.hotkey("win","1")
# pg.hotkey("ctrl","t")
# pg.countdown(2)
# text = ("Wallpaper")
# pg.write(text, interval=0.1)
# pg.press('enter')


#code to auto shutdown the computer.
##1
# pg.hotkey("win","d")
# pg.moveTo(500,500)
# pg.click()
# pg.hotkey("alt","f4")

##2
# pg.moveTo(1101,1753)
# pg.click()
# pg.countdown(2)
# text = ("""cmd""")
# pg.write(text, interval=0.1)
# pg.countdown(2)
# pg.press('enter')
# pg.countdown(2)
# text = ("""shutdown /s""")
# pg.write(text, interval=0.1)
# pg.countdown(2)
# pg.press('enter')
    