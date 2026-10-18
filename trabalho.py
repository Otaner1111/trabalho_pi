import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------------------------------------
# 1. FUNÇÃO PARA CARREGAR IMAGEM JPG
# -------------------------------------------------------------
def carregar_imagem(caminho):
    img = Image.open(caminho).convert('RGB')
    return np.array(img, dtype=np.float32)

# -------------------------------------------------------------
# 2. AS 5 TRANSFORMAÇÕES REQUISITADAS
# -------------------------------------------------------------

# FILTRO 1: Converter para Tons de Cinza
def para_cinza(matriz_rgb):
    if len(matriz_rgb.shape) == 2:
        return matriz_rgb.copy()
    
    r = matriz_rgb[:, :, 0]
    g = matriz_rgb[:, :, 1]
    b = matriz_rgb[:, :, 2]
    return (r + g + b) / 3.0

# FILTRO 2: Blur (Máscara 3x3 de Média)
def aplicar_blur(matriz_cinza):
    lins, cols = matriz_cinza.shape
    img_blur = matriz_cinza.copy()
    m = matriz_cinza
    
    for i in range(1, lins - 1):
        for j in range(1, cols - 1):
            soma = (
                m[i-1, j-1] + m[i-1, j] + m[i-1, j+1] +
                m[i,   j-1] + m[i,   j] + m[i,   j+1] +
                m[i+1, j-1] + m[i+1, j] + m[i+1, j+1]
            )
            img_blur[i, j] = soma / 9.0
            
    return img_blur

# FILTRO 3: Detecção de Bordas (Sobel)
def aplicar_sobel(matriz_cinza):
    lins, cols = matriz_cinza.shape
    sobel_v = np.zeros_like(matriz_cinza)
    sobel_h = np.zeros_like(matriz_cinza)
    m = matriz_cinza

    for i in range(1, lins - 1):
        for j in range(1, cols - 1):
            # Máscara Vertical
            v = ( 1*m[i-1, j-1] +  0*m[i-1, j] + -1*m[i-1, j+1] +
                  2*m[i,   j-1] +  0*m[i,   j] + -2*m[i,   j+1] +
                  1*m[i+1, j-1] +  0*m[i+1, j] + -1*m[i+1, j+1] )
            
            # Máscara Horizontal
            h = ( 1*m[i-1, j-1] +  2*m[i-1, j] +  1*m[i-1, j+1] +
                  0*m[i,   j-1] +  0*m[i,   j] +  0*m[i,   j+1] +
                 -1*m[i+1, j-1] + -2*m[i+1, j] + -1*m[i+1, j+1] )

            sobel_v[i, j] = np.clip(v, 0, 255)
            sobel_h[i, j] = np.clip(h, 0, 255)

    return np.clip(sobel_v + sobel_h, 0, 255)

# FILTRO 4: Unsharp Mask (Original Cinza - Blur)
def aplicar_unsharp(matriz_cinza):
    img_blur = aplicar_blur(matriz_cinza)
    unsharp = matriz_cinza - img_blur
    return np.clip(unsharp, 0, 255)

# FILTRO 5: Sharpening (Original Cinza + Unsharp)
def aplicar_sharpening(matriz_cinza):
    img_unsharp = aplicar_unsharp(matriz_cinza)
    sharpening = matriz_cinza + img_unsharp
    return np.clip(sharpening, 0, 255)

# -------------------------------------------------------------
# 3. PROGRAMA PRINCIPAL
# -------------------------------------------------------------
if __name__ == "__main__":
    # NOME DA SUA IMAGEM (certifique-se de que o arquivo está na mesma pasta!)
    caminho_imagem = "Limes.jpg" 

    try:
        # Carrega a imagem original
        img_original = carregar_imagem(caminho_imagem)
        
        # Aplica as 5 transformações
        img_cinza = para_cinza(img_original)
        img_blur = aplicar_blur(img_cinza)
        img_sobel = aplicar_sobel(img_cinza)
        img_unsharp = aplicar_unsharp(img_cinza)
        img_sharpening = aplicar_sharpening(img_cinza)

        # Monta a exibição em grade
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        axes[0, 0].imshow(img_original.astype(np.uint8))
        axes[0, 0].set_title("Original (Colorida)")
        
        axes[0, 1].imshow(img_cinza, cmap='gray')
        axes[0, 1].set_title("1. Tons de Cinza")
        
        axes[0, 2].imshow(img_blur, cmap='gray')
        axes[0, 2].set_title("2. Blur")
        
        axes[1, 0].imshow(img_sobel, cmap='gray')
        axes[1, 0].set_title("3. Sobel (Bordas)")
        
        axes[1, 1].imshow(img_unsharp, cmap='gray')
        axes[1, 1].set_title("4. Unsharp Mask")
        
        axes[1, 2].imshow(img_sharpening, cmap='gray')
        axes[1, 2].set_title("5. Sharpening")

        for ax in axes.ravel():
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Erro: Não foi possível encontrar a imagem '{caminho_imagem}'. Verifique o nome da imagem na pasta!")