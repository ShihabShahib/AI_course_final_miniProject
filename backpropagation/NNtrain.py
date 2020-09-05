import math


#print("The program is about \nUsing Backpropagation Algorithm in Neural Network\n")
#input("Press enter key to continue")
#print("\nSuppose, in a neural network, Initially there are \nTwo inputs i1=.06 & i2=.12 \nTwo bias b1=.40 & b2=0.55 \nTwo hidden neurons h1,h2 \nTwo target output, O1=0.1 O2=0.9\n")
#print("To design the Neural Network, weights with some random values are given to create a model output\nBut this output will be different from the actual output.There will be error.\nBy doing forward propagation, we can calculate the total error")


i1=.05
i2=.10
target1=0.01
target2=0.99
w1=0.15
w2=.20
w3=.25
w4=.30
w5=.40
w6=.45
w7=.50
w8=.55
b1=.35
b2=.60
n=0.5

#Total net input for h1,h2
neth1=w1*i1+w2*i2+b1*1
#print(neth1)
neth2=w3*i1+w4*i2+b1*1
#print(neth2)

#the output of h1,h2
outh1=1/(1+math.exp(-neth1))
#print(outh1)
outh2=1/(1+math.exp(-neth2))
#print(outh2)

#neto1,o2
neto1=w5*outh1+w6*outh2+b2*1
#print(neto1)
neto2=w7*outh1+w8*outh2+b2*1
#print(neto2)

#outo1,o2
outo1=1/(1+math.exp(-neto1))
#print(outo1)
outo2=1/(1+math.exp(-neto2))
#print(outo2)

#eo1,eo2
eo1=(0.5)*((target1-outo1)**2)
#print(eo1)
eo2=(0.5)*((target2-outo2)**2)
#print(eo2)

#TotalError for the neural network
etotal=eo1+eo2
#print("That total error is ", etotal)

#print("Now our goal is to reduce this error by doing backward propagation.First we will try to change the values of weights and biases")

#Update weight 5
etotal_outo1=(-1)*(target1-outo1)
#print(etotal_outo1)
outo1_neto1=outo1*(1-outo1)
#print(outo1_neto1)
neto1_w5=1*outh1*w5**(1-1)+0+0
#print(neto1_w5)
etotal_w5=etotal_outo1*outo1_neto1*neto1_w5
#print(etotal_w5)
updatedw5=w5-n*etotal_w5
#print(updatedw5)


#Update weight 6
etotal_outo1=(-1)*(target1-outo1)
#print(etotal_outo1)
outo1_neto1=outo1*(1-outo1)
#print(outo1_neto1)
neto1_w6=1*outh2*w6**(1-1)+0+0
#print(neto1_w6)
etotal_w6=etotal_outo1*outo1_neto1*neto1_w6
#print(etotal_w6)
updatedw6=w6-n*etotal_w6
#print(updatedw6)


#Update weight 7
etotal_outo2=(-1)*(target2-outo2)
#print(etotal_outo2)
outo2_neto2=outo2*(1-outo2)
#print(outo2_neto2)
neto2_w7=1*outh1*w7**(1-1)+0+0
#print(neto2_w7)
etotal_w7=etotal_outo2*outo2_neto2*neto2_w7
#print(etotal_w7)
updatedw7=w7-n*etotal_w7
#print(updatedw7)


#Update weight 8
etotal_outo2=(-1)*(target2-outo2)
#print(etotal_outo2)
outo2_neto2=outo2*(1-outo2)
#print(outo2_neto2)
neto2_w8=1*outh2*w8**(1-1)+0+0
#print(neto2_w8)
etotal_w8=etotal_outo2*outo2_neto2*neto2_w8
#print(etotal_w8)
updatedw8=w8-n*etotal_w8
#print(updatedw8)


#update weight 1
eo1_neto1=etotal_outo1*outo1_neto1
#print(eo1_neto1)
neto1_outh1=w5
#print(neto1_outh1)
eo1_outh1=eo1_neto1*neto1_outh1
#print(eo1_outh1)
eo2_neto1=etotal_outo2*outo2_neto2
#print(eo2_neto1)
neto1_outh2=w7
#print(neto1_outh2)
eo2_outh1=eo2_neto1*neto1_outh2
#print(eo2_outh1)
etotal_outh1=eo1_outh1+eo2_outh1
#print(etotal_outh1)
outh1_neth1=outh1*(1-outh1)
#print(outh1_neth1)
neth1_w1=i1
#print(neth1_w1)
etotal_w1=etotal_outh1*outh1_neth1*neth1_w1
#print(etotal_w1)
updatedw1=w1-n*etotal_w1
#print(updatedw1)


#update weight 2
eo1_neto1=etotal_outo1*outo1_neto1
#print(eo1_neto1)
neto1_outh1=w6
#print(neto1_outh1)
eo1_outh11=eo1_neto1*neto1_outh1
#print(eo1_outh11)
eo2_neto2=etotal_outo2*outo2_neto2
#print(eo2_neto2)
neto2_outh2=w8
#print(neto2_outh2)
eo2_outh2=eo2_neto2*neto2_outh2
#print(eo2_outh2)
etotal_outh2=eo1_outh1+eo2_outh2
#print(etotal_outh2)
outh2_neth2=outh2*(1-outh2)
#print(outh2_neth2)
neth2_w2=i1
#print(neth2_w2)
etotal_w2=etotal_outh2*outh2_neth2*neth2_w2
#print(etotal_w2)
updatedw2=w2-n*etotal_w2
#print(updatedw2)



#update weight 3
eo1_outh1=eo1_neto1*neto1_outh1
#print(eo1_outh1)
eo2_neto2=etotal_outo2*outo2_neto2
#print(eo2_neto1)
neto2_outh1=w6
#print(neto2_outh1)
eo2_outh1=eo2_neto2*neto2_outh1
#print(eo2_outh1)
etotal_outh1=eo1_outh1+eo2_outh1
#print(etotal_outh1)
outh1_neth1=outh1*(1-outh1)
#print(outh1_neth1)
neth1_w3=i2
#print(neth1_w1)
etotal_w3=etotal_outh1*outh1_neth1*neth1_w3
#print(etotal_w3)
updatedw3=w3-n*etotal_w3
#print(updatedw3)



#update weight 4
eo1_neto1=etotal_outo1*outo1_neto1
neto1_outh2=w7
eo1_outh2=eo1_neto1*neto1_outh2
#print(eo1_outh2)
eo2_neto2=etotal_outo2*outo2_neto2
#print(eo2_neto2)
neto2_outh2=w8
#print(neto2_outh2)
eo2_outh2=eo2_neto2*neto2_outh2
#print(eo2_outh2)
etotal_outh2=eo1_outh2+eo2_outh2
#print(etotal_outh2)
outh2_neth2=outh2*(1-outh2)
#print(outh2_neth2)
neth2_w4=i2
#print(neth2_w4)
etotal_w4=etotal_outh2*outh2_neth2*neth2_w4
#print(etotal_w4)
updatedw4=w4-n*etotal_w4
#print(updatedw4)


updatedneth1=updatedw1*i1+updatedw2*i2+b1*1
updatedneth2=updatedw3*i1+updatedw4*i2+b1*1
updatedouth1=1/(1+math.exp(-updatedneth1))
updatedouth2=1/(1+math.exp(-updatedneth2))
updatedneto1=updatedw5*updatedouth1+updatedw6*updatedouth2+b2*1
updatedneto2=updatedw7*updatedouth1+updatedw8*updatedouth2+b2*1
updatedouto1=1/(1+math.exp(-updatedneto1))
updatedouto2=1/(1+math.exp(-updatedneto2))
updatedeo1=(0.5)*((target1-updatedouto1)**2)
updatedeo2=(0.5)*((target2-updatedouto2)**2)
#updated Total error
updatedetotal=updatedeo1+updatedeo2
print(updatedetotal)