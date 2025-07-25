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
    void RELU() {
        for (int i = 0; i < filas * columnas; i++) {
            datos[i] = fmaxf(0.0f, datos[i]);
        }
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
    void NormalizarFilas_CUDA();
    Matriz2D MultiplicarCUDA(const Matriz2D& B) const;

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
