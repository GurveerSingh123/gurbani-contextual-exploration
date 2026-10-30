import banidb
import inspect
import json

# print("angs:")
# print(inspect.signature(banidb.angs))
# print(inspect.getdoc(banidb.angs))

# print("\nshabad:")
# print(inspect.signature(banidb.shabad))
# print(inspect.getdoc(banidb.shabad))

# print("\nbanis:")
# print(inspect.signature(banidb.banis))
# print(inspect.getdoc(banidb.banis))

# print("\nsources:")
# print(inspect.signature(banidb.sources))
# print(inspect.getdoc(banidb.sources))

# result = banidb.angs(1)

# print(type(result))
# print(result)


# data = banidb.angs(
#     1,
#     source_id="G",
#     steek=True
# )

# print(json.dumps(
#     data,
#     ensure_ascii=False,
#     indent=2
# ))



# shabad_id = 1

# data = banidb.shabad(shabad_id)

# print(json.dumps(data, ensure_ascii=False, indent=2))

# print(data.keys())

# for key, value in data.items():
#     print("\nKEY:", key)
#     print("VALUE:", value)

# data = banidb.shabad(1111)

# print(json.dumps(data, ensure_ascii=False, indent=2))

# print(data.keys())

# for key, value in data.items():
#     print("\nKEY:", key)
#     print("TYPE:", type(value))
#     print("VALUE:", value if not isinstance(value, (dict, list)) else "")


# print("RAAGS")
# print(banidb.raags())

# print("\nWRITERS")
# print(banidb.writers())

# print("\nBANIS")
# print(banidb.banis())

import banidb

print("=== RAAGS ===")
try:
    result = banidb.raags()
    print(type(result))
    print(result)
except Exception as e:
    print("ERROR:", e)


print("\n=== RAAG 1 ===")
try:
    result = banidb.raag(1)
    print(type(result))
    print(result)
except Exception as e:
    print("ERROR:", e)


print("\n=== WRITERS ===")
try:
    result = banidb.writers()
    print(type(result))
    print(result)
except Exception as e:
    print("ERROR:", e)