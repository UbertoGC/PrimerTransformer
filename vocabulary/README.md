## Acerca del vocabulario

- El vocabulario de GPT2 se almacena en un archivo JSON que asigna 50.257 tokens a sus correspondientes ID de enteros únicos. 
- Incluye caracteres individuales, subpalabras comunes y palabras frecuentes en inglés. 
- Tanto el uso de mayúsculas como el hecho de que hubiera un espacio anterior son importantes. "The" y "the" y " The" se consideran tokens separados dentro del vocabulario de GPT2.

```
{
    "!": 0,
    """: 1,
    ...
    "ĠThe": 383,
    ...
    "The": 464,
    ...
    "the": 1169,
    ...
    "<|endoftext|>": 50256
}
```
### Para qué se usa este vocabulario

- El archivo vocab.json en la tokenización es un diccionario que mapea cada token (que puede ser una palabra, subpalabra o símbolo) a un índice numérico único, por ejemplo: `"hello": 123`

    - Permite convertir tokens de texto a números que el modelo puede procesar. Esto es esencial porque los modelos de lenguaje trabajan con números (vectores), no con texto directamente.

- Permite crear la entrada para el modelo
    - Cuando se tokeniza un texto, se transforma en una secuencia de índices usando vocab.json. Esa secuencia es la que se pasa al modelo para entrenamiento o inferencia.

- Mantiene la consistencia en la codificación y decodificación
    - Ya que se puede usar en sentido inverso para convertir índices de vuelta a tokens y finalmente a texto legible. Esto es clave para que la salida del modelo sea interpretable.

- En conclusión, este archivo funciona como la **"tabla de referencia"** que traduce tokens en números y viceversa, garantizando que la tokenización sea coherente y que el modelo entienda y genere texto correctamente.