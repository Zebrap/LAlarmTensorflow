from keras.models import load_model
import numpy

# Load test dataset
dataset = numpy.loadtxt("test.csv", delimiter=",")  

# Split into input (X) and output (Y) variables
X = dataset[:,0:4]
Y = dataset[:,3]

# Load model
model = load_model("test.model")

# Calculate predictions
predictions = model.predict(X)

f= open("wyniki.csv","w+")
f.write("time,vibre,volume,song,rate\n")
# Print predictions
for i in range(len(Y)):
    print(X[i][0], ",", X[i][1],",", X[i][2],",", X[i][3]," = ", predictions[i][0])
    f.write("%d,%d,%d,%d,%f\n" % (X[i][0],X[i][1],X[i][2],X[i][3],predictions[i][0]))


f.close() 
