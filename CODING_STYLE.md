# Coding Style

---

___basics___

Used `camelCase` except class name using `UpperCase`.  
Used `UPPER` for constexpr.  
Prefere using `struct` instead of `class`..
The declaration of scopes in `struct` or `class` must be done in this order -> public / protected / private.
Initialize default value of members variables in the declaration instead of constructor.
Used `auto` only in loop or long type.  
Used suffixes 64_t, 32_t, 16_t, 8_t instead of basic declarations int, long, char, etc.  
Used `<>` for `#include`.  
```cpp
constexpr uint32_t MY_MACRO{ 12u };

class MyClass
{
public:
    int32_t myVariable{ 0 };

protected:
    void foo() final;

private:
    uint32_t myPrivateVariable{ 0u };

    void myPrivateFoo();
};
```

---

___Condition___  
All time use brackets  
Used const on first element  
```cpp
if (0 == a)
{
    // code
}
```
```cpp
if (1 == a)
{
    // code
}
else if (2 === a)
{
    // code
}
else
{
    // code
}
```

---

___loop___
Define iterator or index in the loop not outside.  
Prefered use `for`.  
```cpp
for (auto& a : list)
{
    // code
}
for (int i{ 0 }; i < value; ++i)
{
    // code
}
```
Used `while` for looping by function returning `boo`, `pointer`.  
```cpp
while (true == isOpen())
{
    // code
}
```