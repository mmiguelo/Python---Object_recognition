import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = '4.jpg' # A imagem que o programa vai analizar

# Função para detectar a cor de cada peça (ROI: Region of Interest)
def detect_color(roi):
    # Definimos os intervalos de cor para vermelho, azul e branco em RGB
    # Importante: Esses valores podem precisar de ser ajustados dependendo da iluminação da imagem.
    red_lower = np.array([0, 0, 100])  # Valor baixo de vermelho no espaço RGB
    red_upper = np.array([100, 100, 255])  # Valor alto de vermelho no espaço RGB

    blue_lower = np.array([100, 0, 0])  # Valor baixo de azul no espaço RGB
    blue_upper = np.array([255, 100, 100])  # Valor alto de azul no espaço RGB

    white_lower = np.array([200, 200, 200])  # Valor baixo de branco no espaço RGB
    white_upper = np.array([255, 255, 255])  # Valor alto de branco no espaço RGB

    # Aqui criamos as máscaras para as cores
    red_mask = cv2.inRange(roi, red_lower, red_upper)
    blue_mask = cv2.inRange(roi, blue_lower, blue_upper)
    white_mask = cv2.inRange(roi, white_lower, white_upper)

    if cv2.countNonZero(red_mask) > 0:
        return "Red"
    elif cv2.countNonZero(blue_mask) > 0:
        return "Blue"
    elif cv2.countNonZero(white_mask) > 0:
        return "White"
    else:
        return "Undefined"

# Função para verificar se a peça é circular
def is_circular(approx):
    # Consideramos uma peça circular se o número de vértices for baixo e a forma for regular
    if len(approx) > 6:  # Um número maior de vértices geralmente indica que a forma não é circular
        return False
    return True

# Função para calcular a área e o perímetro
def get_area_and_perimeter(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return area, perimeter

# Função para desenhar a localização e centro de gravidade sobre a imagem
def draw_contours_and_center(img, contours):
    for contour in contours:
        # Encontrar a caixa delimitadora e o centro de massa (centro de gravidade)
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)

        # Desenhar a caixa delimitadora
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Desenhar o centro de gravidade
        cv2.circle(img, center, 5, (0, 0, 255), -1)

        # Detectar a cor
        roi = img[y:y + h, x:x + w]
        color = detect_color(roi)

        # Escrever o tipo e características sobre a peça
        cv2.putText(img, f"Color: {color}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        shape = "Circular" if is_circular(contour) else "Non-circular"
        cv2.putText(img, f"Shape: {shape}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Processar a imagem
img = cv2.imread(image_path)
if img is None:
    print(f"Erro: Nao foi possivel encontrar {image_path}")
else:
    # Convertemos para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicamos thresholding para segmentação
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Encontramos os contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_pieces = 0
    circular_pieces = 0
    non_circular_pieces = 0
    red_pieces = 0
    blue_pieces = 0
    white_pieces = 0
    pieces_with_holes = 0
    pieces_without_holes = 0
    area_per_piece = []

    # Aqui Processamos os contornos
    for contour in contours:
        area, perimeter = get_area_and_perimeter(contour)
        total_pieces += 1
        
        # Verificamos a cor
        x, y, w, h = cv2.boundingRect(contour)
        roi = img[y:y + h, x:x + w]
        color = detect_color(roi)
        if color == "Red":
            red_pieces += 1
        elif color == "Blue":
            blue_pieces += 1
        elif color == "White":
            white_pieces += 1
        
        # Verificamos a forma
        if is_circular(contour):
            circular_pieces += 1
        else:
            non_circular_pieces += 1
        
        # Contamos as peças com furos
        # Aqui assumimos que as peças com um buraco possuem área de contorno menor)
        if area < 1000:  # Ajustar este valor conforme necessário
            pieces_with_holes += 1
        else:
            pieces_without_holes += 1
        
        area_per_piece.append(area)

    # Imprimir os resultados
    print(f"Imagem {image_path}:")
    print(f"Total de peças: {total_pieces}")
    print(f"Peças vermelhas: {red_pieces}")
    print(f"Peças azuis: {blue_pieces}")
    print(f"Peças brancas: {white_pieces}")
    print(f"Peças circulares: {circular_pieces}")
    print(f"Peças não circulares: {non_circular_pieces}")
    print(f"Peças com furos: {pieces_with_holes}")
    print(f"Peças sem furos: {pieces_without_holes}")
    
    # Peça com maior e menor área
    print(f"Maior área: {max(area_per_piece)} pixels")
    print(f"Menor área: {min(area_per_piece)} pixels\n")
    
    # Desenhar a localização e centro de gravidade
    draw_contours_and_center(img, contours)
    
    # Exibir a imagem com as anotações
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Imagem {image_path}")
    plt.show()
