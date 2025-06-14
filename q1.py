from sklearn.linear_model import Perceptron

# Representação dos caracteres H e T
h = [1,1,1,0,1,0,0,1,0]
t = [1,0,1,1,1,1,1,0,1]
#T e H distorcidos
tDist = [1,0,0,1,1,1,1,0,0]
hDist = [1,1,1,0,1,0,1,1,1]


x = [t, h, tDist, hDist]
y = [0, 1, 0, 1]

modelo = Perceptron(max_iter=1000, eta0=0.4)
modelo.fit(x,y)

tDist2 = [0,0,1,1,1,1,0,0,1]
hDist2 = [1,1,1,0,0,0,1,1,0]

tDist3 = [0,1,0,0,1,0,1,1,1]
hDist3 = [1,1,1,0,1,0,1,1,0]

print("T:", modelo.predict([t]))
print("H:", modelo.predict([h]))
print("T Dist:", modelo.predict([tDist2]))
print("H Dist:", modelo.predict([hDist2]))
print("T Dist:", modelo.predict([tDist3]))
print("H Dist:", modelo.predict([hDist3]))

#A depender da distorção ele consegue indentificar, porém a casos como um t de cabeça para 
#baixo ele não consegue identificar