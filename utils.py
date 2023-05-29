
def showGraphic(history):
    import matplotlib.pyplot as plt
    
    # plotar a perda de treinamento e validação ao longo do tempo
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perda do Modelo')
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend(['Treinamento', 'Validação'], loc='upper right')
    plt.show()

    # plotar a precisão de treinamento e validação ao longo do tempo
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Precisão do Modelo')
    plt.ylabel('Precisão')
    plt.xlabel('Época')
    plt.legend(['Treinamento', 'Validação'], loc='lower right')
    plt.show()


def predictList(function, directory, show=True):
    import os
    result = [[function(f"{directory}/{item}"), f"File: {item}"] for item in os.listdir(directory)]
    if show:
        for classe, file in result:
            print(f"Classe : {classe} ----- File: {file}")
    return result


