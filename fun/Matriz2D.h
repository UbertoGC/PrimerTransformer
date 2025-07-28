#ifndef MATRIZ2D_H
#define MATRIZ2D_H

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; \
        std::cerr << " code=" << err << " " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

class Matriz2D {
private:
    int filas;
    int columnas;
    float* datos;

public:
    // =====================
    // Constructores
    // =====================
    
    Matriz2D() : filas(0), columnas(0), datos(nullptr) {}

    Matriz2D(int f, int c, int tipo = 0) : filas(f), columnas(c) {
        size_t size = f * c;
        datos = new float[size];

        if (tipo == 1) { // Aleatorio [0.0, 9.9]
            srand(time(NULL));
            for (size_t i = 0; i < size; i++) {
                datos[i] = (rand() % 100) / 10.0f;
            }
        }
        else if (tipo == 2) { // Aleatorio [-1.0, 1.0]
            srand(time(NULL));
            for (size_t i = 0; i < size; i++) {
                datos[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            }
        }
        else { // Ceros
            memset(datos, 0, size * sizeof(float));
        }
    }

    Matriz2D(const Matriz2D& other) : filas(other.filas), columnas(other.columnas) {
        size_t size = filas * columnas;
        datos = new float[size];
        memcpy(datos, other.datos, size * sizeof(float));
    }

    ~Matriz2D() {
        delete[] datos;
    }

    // =====================
    // Operadores
    // =====================
    float& operator()(int i, int j) {
        return datos[i * columnas + j];
    }

    const float& operator()(int i, int j) const {
        return datos[i * columnas + j];
    }

    // =====================
    // Métodos básicos
    // =====================
    int fil() const { return filas; }
    int col() const { return columnas; }
    float* operator[](int fila) {
        return &datos[fila * columnas];
    }

    const float* operator[](int fila) const {
        return &datos[fila * columnas];
    }

    void ReSize(int f, int c) {
        delete[] datos;
        filas = f;
        columnas = c;
        datos = new float[f * c];
        memset(datos, 0, f * c * sizeof(float));
    }
    Matriz2D& operator=(const Matriz2D& other) {
        if (this == &other) return *this;
        if (filas * columnas != other.filas * other.columnas) {
            delete[] datos;
            filas = other.filas;
            columnas = other.columnas;
            datos = new float[filas * columnas];
        }
        memcpy(datos, other.datos, filas * columnas * sizeof(float));
        return *this;
    }


    void Random() {
        srand(time(NULL));
        for (int i = 0; i < filas * columnas; i++) {
            datos[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        }
    }
    // =====================
    // Operación de Broadcast
    // =====================
    void SumarFila(const Matriz2D& filaBias) {
        if (filaBias.fil() != 1 || filaBias.col() != columnas) {
            throw std::runtime_error("Error: bias debe ser [1 x columnas]");
        }
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                datos[i * columnas + j] += filaBias(0, j);
            }
        }
    }

    // =====================
    // Funciones CPU
    // =====================
    
    Matriz2D Transpuesta() const {
        Matriz2D T(columnas, filas); // Invertimos dimensiones
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                T(j, i) = (*this)(i, j);
            }
        }
        return T;
    }
    void Matriz2D::CopiarMatrizDatos(int pos_x, int pos_y, const Matriz2D& B) {
        if (pos_x < 0 || (pos_x + B.filas) > filas || pos_y < 0 || (pos_y + B.columnas) > columnas) {
            std::cerr << "Error: Posicion o dimensiones invalidas." << std::endl;
            return;
        }
        #pragma omp parallel for
        for (int i = 0; i < B.filas; i++){
            for (int j = 0; j < B.columnas; j++){
                (*this)(pos_x + i,pos_y + j) = B(i,j);
            }
        }
    }
    void Matriz2D::NormalizarFilas(const Matriz2D& gamma, const Matriz2D& beta) {
        const double epsilon = 1e-5;

        if (gamma.col() != columnas || beta.col() != columnas) {
            throw std::runtime_error("Dimensiones de gamma/beta no coinciden con columnas");
        }

        #pragma omp parallel for
        for (int i = 0; i < filas; i++) {
            double suma = 0.0, suma_cuadrados = 0.0;

            // 1. Calcular media
            for (int j = 0; j < columnas; j++) {
                suma += (*this)(i, j);
            }
            double media = suma / columnas;

            // 2. Calcular varianza
            for (int j = 0; j < columnas; j++) {
                double diff = (*this)(i, j) - media;
                suma_cuadrados += diff * diff;
            }
            double varianza = suma_cuadrados / columnas;
            double std_dev = sqrt(varianza + epsilon);

            // 3. Normalizar y aplicar gamma, beta
            #pragma omp simd
            for (int j = 0; j < columnas; j++) {
                double norm = ((*this)(i, j) - media) / std_dev;
                (*this)(i, j) = norm * gamma(0, j) + beta(0, j);
            }
        }
    }



    void RELU() {
        for (int i = 0; i < filas * columnas; i++) {
            datos[i] = fmaxf(0.0f, datos[i]);
        }
    }
    void Matriz2D::Fill(float valor) {
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                (*this)(i,j) = valor;
            }
        }
    }

    Matriz2D MultiplicarCPU(const Matriz2D& B) const {
        if (columnas != B.filas) {
            throw std::runtime_error("Dimensiones incompatibles en MultiplicarCPU");
        }
        Matriz2D C(filas, B.columnas);
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < B.columnas; j++) {
                float suma = 0.0f;
                for (int k = 0; k < columnas; k++) {
                    suma += (*this)(i, k) * B(k, j);
                }
                C(i, j) = suma;
            }
        }
        return C;
    }

    void SoftmaxFilas() {
        for (int i = 0; i < filas; i++) {
            float maxVal = (*this)(i, 0);
            for (int j = 1; j < columnas; j++) {
                if ((*this)(i, j) > maxVal) maxVal = (*this)(i, j);
            }

            float sum = 0.0f;
            for (int j = 0; j < columnas; j++) {
                (*this)(i, j) = expf((*this)(i, j) - maxVal);
                sum += (*this)(i, j);
            }

            for (int j = 0; j < columnas; j++) {
                (*this)(i, j) /= sum;
            }
        }
    }
    
    // =====================
    // Funciones CUDA
    // =====================
    void RELU_CUDA();
    void SoftmaxFilas_CUDA();
    //void NormalizarFilas_CUDA();
    void Matriz2D::NormalizarFilas_CUDA(const Matriz2D& gamma, const Matriz2D& beta);
    Matriz2D MultiplicarCUDA(const Matriz2D& B) const;
    void Matriz2D::SumarFilaCUDA(const Matriz2D& fila);//Sumar una sola fila (vector) a todas las filas de la matriz. bias o braodcasting
    void SumarMatrizCUDA(const Matriz2D& otra);//Sumar dos matrices completas elemento por elemento (misma dimensión).
    void Matriz2D::EscalarCUDA(float escalar);//Matriz por un escalar (multiplicación por un escalar).
    // =====================
    // Métodos de acceso
    // =====================
    int Filas() const { return filas; }
    int Columnas() const { return columnas; }
    float* Datos() const { return datos; }

    // =====================
    // Salida
    // =====================
    friend std::ostream& operator<<(std::ostream& os, const Matriz2D& mat) {
        for (int i = 0; i < mat.filas; i++) {
            for (int j = 0; j < mat.columnas; j++) {
                os << std::setw(10) << mat(i, j) << " ";
            }
            os << "\n";
        }
        return os;
    }
};

#endif // MATRIZ2D_H
