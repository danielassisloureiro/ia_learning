import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Dropdown

# Funções de ativação
def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def linear(x):
    return x

def step(x):
    return 1 if x >= 0 else 0

# Função principal
def simular_neuronio(x1, x2, w1, w2, bias, ativacao):
    z = x1 * w1 + x2 * w2 + bias

    # Seleciona função de ativação
    if ativacao == 'ReLU':
        y = relu(z)
    elif ativacao == 'Sigmoid':
        y = sigmoid(z)
    elif ativacao == 'Tanh':
        y = tanh(z)
    elif ativacao == 'Linear':
        y = linear(z)
    elif ativacao == 'Step':
        y = step(z)
    else:
        y = z

    # Print dos resultados
    print(f"Soma ponderada (z): {z:.4f}")
    print(f"Saída com {ativacao}: {y:.4f}")

    # Gráfico
    plt.figure(figsize=(6, 3))
    plt.bar(['z (soma)', f'y ({ativacao})'], [z, y], color=['gray', 'green'])
    plt.title(f"Resultado do Neurônio com {ativacao}")
    plt.ylim(-2, 2 if ativacao != 'ReLU' else max(2, y + 0.5))
    plt.grid(True)
    plt.show()

# Interface interativa
interact(
    simular_neuronio,
    x1=FloatSlider(value=1.0, min=-5.0, max=5.0, step=0.1, description='x1'),
    x2=FloatSlider(value=2.0, min=-5.0, max=5.0, step=0.1, description='x2'),
    w1=FloatSlider(value=0.5, min=-5.0, max=5.0, step=0.1, description='w1'),
    w2=FloatSlider(value=-1.0, min=-5.0, max=5.0, step=0.1, description='w2'),
    bias=FloatSlider(value=0.0, min=-5.0, max=5.0, step=0.1, description='bias'),
    ativacao=Dropdown(
        options=['ReLU', 'Sigmoid', 'Tanh', 'Linear', 'Step'],
        value='ReLU',
        description='Ativação'
    )
)

#Funções de ativação suportadas
#ReLU	Linear positiva (desativa valores negativos)	[0, ∞)
#Sigmoid	Curva em S suave, útil para probabilidades	(0, 1)
#Tanh	Curva em S centrada em 0	(-1, 1)
#Linear	Sem ativação — saída igual à soma z	ℝ
#Step	Ativação binária (ativado ou não)	{0, 1}