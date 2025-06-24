#ifndef __MATRIZ2D_H__
#define __MATRIZ2D_H__
#include <random>
#include <omp.h>
#include <iostream>
class Vector2D{
private:
    double* v;
    double largo;
public:
    friend class Matriz2D;
    Vector2D();
    Vector2D(int);
    int lar();
    double& operator[](int);
    Vector2D& operator<<(const Vector2D&);
    Vector2D& operator+=(const Vector2D&);
    ~Vector2D();
};
Vector2D::Vector2D(){
    largo = 0;
    v = nullptr;
}
Vector2D::Vector2D(int x){
    largo = x;
    #pragma omp parallel for
    for (int i = 0; i < largo; ++i) {
        v[i] = 0;
    }
}
int Vector2D::lar(){
    return largo;
}
double& Vector2D::operator[](int i){
    return v[i];
}
Vector2D& Vector2D::operator<<(const Vector2D& B) {
    if (this == &B) 
        return *this;
    int minimo = std::min(this->largo, B.largo);
    #pragma omp parallel for
    for (int i = 0; i < minimo; ++i) {
        this->v[i] = B.v[i];
    }
    return *this;
}
Vector2D& Vector2D::operator+=(const Vector2D& B) {
    int minimo = std::min(this->largo, B.largo);
    #pragma omp parallel for
    for (int i = 0; i < minimo; ++i) {
        this->v[i] += B.v[i];
    }
    return *this;
}
Vector2D::~Vector2D(){
    delete[] v;
}

class Matriz2D
{
private:
    int alto;
    int ancho;
    double** m;
    Vector2D* vectores;
    void Inicializar(int, int);
    void Zero();
public:
    Matriz2D();
    Matriz2D(int, int);
    Matriz2D(int, int, double**);
    Matriz2D Transpuesta();
    void Transponer();
    void Relacionar(const Matriz2D&);
    void ReSize(int, int);
    void Limpiar();
    void Random();
    int fil();
    int col();
    Vector2D& operator[](int);
    Matriz2D& operator=(const Matriz2D&);
    friend std::ostream& operator<<(std::ostream&, const Matriz2D&);
    ~Matriz2D();
};
Matriz2D::Matriz2D(){
    vectores = nullptr;
    alto = 0;
    ancho = 0;
    m = nullptr;
}
Matriz2D::Matriz2D(int x, int y){
    this->Inicializar(x, y);
    this->Zero();
}
Matriz2D::Matriz2D(int x, int y, double**data)
{
    this->Inicializar(x,y);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < alto; i++){
        for (int j = 0; j < ancho; j++){
            m[i][j] = data[i][j];
        }
    }
}
void Matriz2D::Inicializar(int x, int y){
    alto = x;
    ancho = y;
    m = new double*[alto];
    vectores = new Vector2D[alto];
    #pragma omp parallel for
    for (int i = 0; i < alto; i++){
        m[i] = new double[ancho];
        vectores[i].v = m[i];
        vectores[i].largo = ancho;
    }
}
void Matriz2D::Zero(){
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < alto; i++){
        for (int j = 0; j < ancho; j++){
            m[i][j] = 0;
        }
    }
}
Matriz2D Matriz2D::Transpuesta(){
    Matriz2D t(ancho, alto);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < alto; i++){
        for (int j = 0; j < ancho; j++){
            t[j][i] = m[i][j];
        }
    }
    return t;
}
void Matriz2D::Transponer(){
    double **tmp = m;
    m = nullptr;
    #pragma omp parallel for
    for (int i = 0; i < ancho; i++){
        vectores[i].v = nullptr;
    }
    delete[] vectores;
    this->Inicializar(ancho, alto);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < alto; i++){
        for (int j = 0; j < ancho; j++){
            m[i][j] = tmp[j][i];
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < alto; i++){
        delete[] tmp[i];
    }
    delete[] vectores;
    delete[] tmp;
}
void Matriz2D::Relacionar(const Matriz2D& B){
    this->Limpiar();
    this->alto = B.alto;
    this->ancho = B.ancho;
    this->m = B.m;
}
void Matriz2D::ReSize(int x, int y){
    if(x != alto || y != ancho){
        this->Limpiar();
        this->Inicializar(x, y);
    }
    this->Zero();
}
void Matriz2D::Limpiar(){
    if(m == nullptr)
        return;
    for (int i = 0; i < alto; i++) {
        delete[] m[i];
        
    }
    delete[] vectores;
    delete[] m;
    alto = 0;
    ancho = 0;
    m = nullptr;
    vectores = nullptr;
}
void Matriz2D::Random(){
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < alto; i++){
        for (int j = 0; j < ancho; j++){
            std::mt19937 gen(i * 100 + j);
            std::uniform_real_distribution<double> dist(0.5, 1.5);
            m[i][j] = dist(gen);
        }
    }
}
int Matriz2D::fil(){
    return alto;
}
int Matriz2D::col(){
    return ancho;
}
Vector2D& Matriz2D::operator[](int i){
    return vectores[i];
}
Matriz2D& Matriz2D::operator=(const Matriz2D& B) {
    if (this == &B) 
        return *this;
    if(ancho != B.ancho || alto != B.alto){
        Limpiar();
        this->Inicializar(B.alto, B.ancho);
    }
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < alto; ++i) {
        for (int j = 0; j < ancho; ++j){
            m[i][j] = B.m[i][j];
        }
    }
    return *this;
}
std::ostream& operator<<(std::ostream& os, const Matriz2D& A){
    for (int i = 0; i < A.alto; i++){
        os<<'[';
        for (int j = 0; j < A.ancho; j++){
            os<<A.m[i][j];
            if(j != (A.ancho - 1)){
                os<<", ";
            }
        }
        os<<"]\n";
    }
    return os;
}
Matriz2D::~Matriz2D()
{
    Limpiar();
}

#endif