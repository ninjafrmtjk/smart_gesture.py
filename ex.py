import uuid

# Генерируем случайный уникальный идентификатор (версия 4)
my_unique_id = uuid.uuid4()
my_unique_id2 = uuid.uuid4()

if my_unique_id == my_unique_id2:
    print("Уникальные идентификаторы совпали, что крайне маловероятно.")
else:
    print("Уникальные идентификаторы различны.")
    print(my_unique_id, my_unique_id2) 


# Вывод: что-то вроде 550e8400-e29b-41d4-a716-446655440000