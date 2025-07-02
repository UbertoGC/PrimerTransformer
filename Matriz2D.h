#ifndef __MATRIZ2D_H__
#define __MATRIZ2D_H__
#include <random>
#include <omp.h>
#include <iostream>
class Vector2D{
private:
    double* v;
    int largo;
public:
    friend class Matriz2D;
    Vector2D();
    Vector2D(int);
    void Random();
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
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < largo; ++i) {
        v[i] = 0;
    }
}
void Vector2D::Random(){
    v = new double[largo];
    for (int i = 0; i < largo; ++i) {
        std::mt19937 gen(i * 10);
        std::uniform_real_distribution<double> dist(0.5, 2.5);
        v[i] = dist(gen);
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
    Matriz2D(Matriz2D&);
    Matriz2D(int, int, int t = 2);
    Matriz2D(int, int, double**);
    Matriz2D Transpuesta();
    void Transponer();
    void Relacionar(const Matriz2D&);
    void ReSize(int, int);
    void Limpiar();
    void Random();
    void NormalizarFilas();
    void SoftmaxFilas();
    void CopiarMatrizDatos(int, int, const Matriz2D&);
    int fil();
    int col();
    Vector2D& operator[](int);
    Matriz2D& operator=(const Matriz2D&);
    friend Matriz2D operator+(const Matriz2D&, const Matriz2D&);
    friend Matriz2D operator*(const Matriz2D&, const Matriz2D&);
    Matriz2D& operator*=(const double&);
    Matriz2D& operator+=(const double&);
    Matriz2D& operator+=(const Matriz2D&);
    friend std::ostream& operator<<(std::ostream&, const Matriz2D&);
    ~Matriz2D();
};
Matriz2D::Matriz2D(){
    vectores = nullptr;
    alto = 0;
    ancho = 0;
    m = nullptr;
}
Matriz2D::Matriz2D(Matriz2D& B){
    this->Inicializar(B.alto, B.ancho);
    #pragma omp parallel for
    for (int i = 0; i < alto; i++){
        #pragma omp simd
        for (int j = 0; j < ancho; j++){
            m[i][j] = B.m[i][j];
        }
    }
}
Matriz2D::Matriz2D(int x, int y, int t){
    if(t == 0){
        this->Inicializar(x, y);
        for (int i = 0; i < alto; i++){
            for (int j = 0; j < ancho; j++){
                if(j <= i){
                    m[i][j] = 0.0;
                }
                else{
                    m[i][j] = -1e9;
                }
            }
        }
    }
    if(t == 1){
        this->Inicializar(x, y);
        this->Random();
    }
    else{
        this->Inicializar(x, y);
        this->Zero();
    }
}
Matriz2D::Matriz2D(int x, int y, double**data)
{
    this->Inicializar(x,y);
    #pragma omp parallel for
    for (int i = 0; i < alto; i++){
        #pragma omp simd
        for (int j = 0; j < ancho; j++){
            m[i][j] = data[i][j];
        }
    }
}
void Matriz2D::Inicializar(int x, int y){
    this->alto = x;
    this->ancho = y;
    this->m = new double*[alto];
    this->vectores = new Vector2D[alto];
    int i = 0;
    int i_sig = i + 32;
    while (i < alto){
        #pragma omp parallel for
        for (int k = i; k < i_sig && k < alto; k++){
            this->m[k] = new double[ancho];
            this->vectores[k].v = this->m[k];
            this->vectores[k].largo = this->ancho;
        }
        i = i_sig;
        i_sig += 32;
    }
}
void Matriz2D::Zero(){
    #pragma omp parallel for
    for (int i = 0; i < alto; i++){
        #pragma omp simd
        for (int j = 0; j < ancho; j++){
            m[i][j] = 0;
        }
    }
}
Matriz2D Matriz2D::Transpuesta(){
    Matriz2D t(ancho, alto);
    #pragma omp parallel for
    for (int i = 0; i < alto; i++){
        #pragma omp simd
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
    #pragma omp parallel for
    for (int i = 0; i < alto; i++){
        #pragma omp simd
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
    int i = 0;
    int i_sig = i + 64;
    while (i < alto) {
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int k = i; k < i_sig && k < alto; k++) {
            for (int j = 0; j < ancho; j++){
                std::minstd_rand rng(k+j);
                std::uniform_real_distribution<double> dist(0.5, 2.5);
                m[k][j] = dist(rng);
            }
        }
        i = i_sig;
        i_sig += 64;
    }
}
void Matriz2D::NormalizarFilas(){
    #pragma omp parallel for
    for (int i = 0; i < alto; i++) {
        double sum = 0.0, var = 0.0;
        for (int j = 0; j < ancho; j++) {
            sum += m[i][j];
        }
        for (int j = 0; j < ancho; j++) {
            var += (m[i][j] - sum / ancho) * (m[i][j] - sum / ancho);
        }
        if (var != 0) {
            #pragma omp simd
            for (int j = 0; j < ancho; j++) {
                m[i][j] = m[i][j] - sum / var;
            }
        }
    }
}
void Matriz2D::SoftmaxFilas() {
    #pragma omp parallel for
    for (int i = 0; i < alto; i++) {
        double max_val = m[i][0];
        for (int j = 1; j < ancho; j++) {
            if (m[i][j] > max_val) {
                max_val = m[i][j];
            }
        }
        double sum_exp = 0.0;
        for (int j = 0; j < ancho; j++) {
            m[i][j] = exp(m[i][j] - max_val);
            sum_exp += m[i][j];
        }
        for (int j = 0; j < ancho; j++) {
            m[i][j] /= sum_exp;
        }
    }
}
void Matriz2D::CopiarMatrizDatos(int pos_x, int pos_y, const Matriz2D& B) {
    if (pos_x < 0 || (pos_x + B.alto) > alto || pos_y < 0 || (pos_y + B.ancho) > ancho) {
        std::cerr << "Error: Posicion o dimensiones invalidas." << std::endl;
        return;
    }
    #pragma omp parallel for
    for (int i = 0; i < B.alto; i++){
        for (int j = 0; j < B.ancho; j++){
            m[pos_x + i][pos_y + j] = B.m[i][j];
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
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < alto; ++i) {
        for (int j = 0; j < ancho; ++j){
            m[i][j] = B.m[i][j];
        }
    }
    return *this;
}
Matriz2D operator*(const Matriz2D& A, const Matriz2D& B) {
    if (A.ancho != B.alto) {
        std::cerr << "Error: Las matrices no son compatibles para la multiplicación." << std::endl;
        return Matriz2D();
    }
    Matriz2D C(A.alto, B.ancho);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < A.alto; ++i) {
        for (int j = 0; j < B.ancho; ++j) {
            C.m[i][j] = 0;
            for (int k = 0; k < A.ancho; ++k) {
                C.m[i][j] += A.m[i][k] * B.m[k][j];
            }
        }
    }
    return C;
}
Matriz2D operator+(const Matriz2D& A, const Matriz2D& B) {
    if (A.alto != B.alto || A.ancho != B.ancho) {
        std::cerr << "Error: Las matrices no son compatibles para la suma." << std::endl;
        return Matriz2D();
    }
    Matriz2D C(A.alto, B.ancho);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < A.alto; ++i) {
        for (int j = 0; j < B.ancho; ++j) {
            C.m[i][j] = A.m[i][j] + B.m[i][j];
        }
    }
    return C;
}
Matriz2D& Matriz2D::operator*=(const double& escala) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < alto; ++i) {
        for (int j = 0; j < ancho; ++j) {
            m[i][j] *= escala;
        }
    }
    return *this;
}
Matriz2D& Matriz2D::operator+=(const double& valor) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < alto; ++i) {
        for (int j = 0; j < ancho; ++j) {
            m[i][j] += valor;
        }
    }
    return *this;
}
Matriz2D& Matriz2D::operator+=(const Matriz2D& B) {
    if (alto != B.alto || ancho != B.ancho) {
        std::cerr << "Error: Las matrices no son compatibles para la suma." << std::endl;
        return *this;
    }
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < alto; ++i) {
        for (int j = 0; j < ancho; ++j) {
            m[i][j] += B.m[i][j];
        }
    }
    return *this;
}
std::ostream& operator<<(std::ostream& os, const Matriz2D& A){
    os<<"["<<A.alto<<" x "<<A.ancho<<"]\n";
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
Matriz2D::~Matriz2D() {
    for (int i = 0; i < alto; ++i)
        delete[] m[i];
    delete[] m;
}

#endif