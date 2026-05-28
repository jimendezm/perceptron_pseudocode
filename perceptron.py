import math

degrees_fahrenheit = [
    -20.0000, -8.4211, 3.1579, 14.7368, 26.3158,
    37.8947, 49.4737, 61.0526, 72.6316, 84.2105,
    95.7895, 107.3684, 118.9474, 130.5263, 142.1053,
    153.6842, 165.2632, 176.8421, 188.4211, 200.0000
]

"""
FUNCTION fahrenheit_to_celsius(f):
    RETURN (f - 32) * 5 / 9
"""
def fahrenheit_to_celsius(f):
    return (f - 32) * 5 / 9

degrees_celsius = []
for degreeF in degrees_fahrenheit:
    degrees_celsius+=[fahrenheit_to_celsius(degreeF)]

# Implement the pseudocode


"""
FUNCTION mean(elements):
    IF elements is empty:
        RAISE error "Cannot compute mean of an empty list"

    total ← 0

    FOR each element in elements:
        total ← total + element

    RETURN total / number of elements
"""
def mean(elements):
    if elements==[]:
        raise ValueError("Cannot compute mean of an empty list")
    
    total=0
    
    for element in elements:
        total=total+element
    
    return total/len(elements)

""""
FUNCTION standard_deviation(elements):
    IF elements is empty:
        RAISE error "Cannot compute standard deviation of an empty list"

    mean_value ← mean(elements)
    variance ← average of (element - mean_value)^2 for each element

    RETURN square_root(variance)
"""
def standard_deviation(elements):
    if elements==[]:
        raise ValueError("Cannot compute standard deviation of an empty list")
    
    mean_value=mean(elements)
    variance = sum((element-mean_value)**2 for element in elements) / len(elements)
    return math.sqrt(variance)


"""
FUNCTION normalize(dataset):
    value_hat ← mean(dataset)
    value_sd  ← standard_deviation(dataset)

    IF value_sd == 0:
        RAISE error "Cannot normalize values with zero standard deviation"

    value_norm ← empty list

    FOR each element in dataset:
        value ← (element - value_hat) / value_sd
        value_norm.append(value)

    RETURN value_norm
"""
def normalize(dataset):
    value_hat=mean(dataset)
    value_sd=standard_deviation(dataset)
    
    if value_sd==0:
        raise ValueError("Cannot normalize values with zero standard deviation")
    
    value_norm=[]
    
    for element in dataset:
        value=(element-value_hat)/value_sd
        value_norm.append(value)
        
    return value_norm

"""
FUNCTION denormalize(value, dataset):
    value_hat ← mean(dataset)
    value_sd  ← standard_deviation(dataset)

    // Inverse of normalization:
    // original_value = normalized_value * standard_deviation + mean
    RETURN value * value_sd + value_hat
"""
def denormalize(value,dataset):
    value_hat=mean(dataset)
    value_sd=standard_deviation(dataset)
    
    return value*value_sd+value_hat

x_normalized = normalize(degrees_fahrenheit)
y_normalized = normalize(degrees_celsius)

"""
FUNCTION perceptron(x, w, b):
    // Linear model / single-input perceptron
    // y_hat = w * x + b
    RETURN w * x + b
"""
def perceptron(x,w,b):
    return w*x+b

"""
FUNCTION train_perceptron(x_values, y_values, w, b, learning_rate, epochs):
    IF length of x_values != length of y_values:
        RAISE error "x_values and y_values must have the same length"

    IF x_values is empty:
        RAISE error "Cannot train with an empty dataset"

    n ← length of x_values

    training_history ← {
        epochs:  empty list,
        weights: empty list,
        biases:  empty list,
        mse:     empty list
    }

    FOR epoch FROM 0 TO epochs - 1:
        dw  ← 0
        db  ← 0
        mse ← 0

        FOR each pair (x, y) from x_values and y_values:
            // Predict output with the current weight and bias
            y_hat ← perceptron(x, w, b)

            // Error / residual
            error ← y - y_hat

            // Mean Squared Error contribution
            mse ← mse + error^2

            // Accumulate gradient terms
            dw ← dw + x * error
            db ← db + error

        // Final MSE for this epoch
        mse ← mse / n

        // Partial derivative of MSE with respect to w
        d_mse_dw ← (-2 / n) * dw

        // Partial derivative of MSE with respect to b
        d_mse_db ← (-2 / n) * db

        // Gradient descent update rule
        w ← w - learning_rate * d_mse_dw
        b ← b - learning_rate * d_mse_db

        training_history["epochs"].append(epoch + 1)
        training_history["weights"].append(w)
        training_history["biases"].append(b)
        training_history["mse"].append(mse)

        PRINT "Epoch", epoch + 1, "w=", w, "b=", b, "mse=", mse

    RETURN w, b, training_history
"""
def train_perceptron(x_values, y_values, w, b, learning_rate, epochs):
    if len(x_values)!=len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    
    if x_values==[]:
        raise ValueError("Cannot train with an empty dataset")
    
    n=len(x_values)
    
    training_history={
        "epochs":[],
        "weights":[],
        "biases":[],
        "mse":[]
    }
    
    for epoch in range(epochs):
        dw=0
        db=0
        mse=0
        
        for x,y in zip(x_values,y_values):
            y_hat=perceptron(x,w,b)
            
            error=y-y_hat
            
            mse=mse+error**2
            
            dw=dw+x*error
            db=db+error
        mse=mse/n
        
        d_mse_dw=(-2/n)*dw
        d_mse_db=(-2/n)*db
        
        w=w-learning_rate*d_mse_dw
        b=b-learning_rate*d_mse_db
        
        training_history["epochs"].append(epoch+1)
        training_history["weights"].append(w)
        training_history["biases"].append(b)
        training_history["mse"].append(mse)
        
        print("Epoch",epoch+1,"w=", w, "b=", b, "mse=", mse)
        
    return w,b,training_history

def main():
    w=0
    b=0
    
    learning_rate=0.3
    epochs=8
    
    mean_value=mean(x_normalized)
    sd_value=standard_deviation(x_normalized)
    
    print(mean_value,sd_value)
    
    w, b, training_history = train_perceptron(
        x_normalized,
        y_normalized,
        w,
        b,
        learning_rate,
        epochs
    )
    
    print("Final weight:",w)
    print("Final bias:",b)
    
    fahrenheit_value = -20
    fahrenheit_normalized = (
        fahrenheit_value - mean(degrees_fahrenheit)
    ) / standard_deviation(degrees_fahrenheit)

    predicted_celsius_normalized = perceptron(fahrenheit_normalized, w, b)
    predicted_celsius = denormalize(predicted_celsius_normalized, degrees_celsius)
    
    print("Predicted normalized Celsius: ", predicted_celsius_normalized)
    print("Predicted Celsius: ", predicted_celsius)
 

if __name__ == "__main__":
    main()          