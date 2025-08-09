import casino as cs


def test_pow():
    x = cs.Constant(3)
    y = cs.Constant(2)
    z = x ** y
    assert z.sample() == 3**2

def test_add():
    x = cs.Constant(1)
    y = cs.Constant(2)
    z = x + y
    assert z.sample() == 3

def test_sub():
    x = cs.Constant(1)
    y = cs.Constant(2)
    z = x - y
    assert z.sample() == -1

def test_mul():
    x = cs.Constant(1)
    y = cs.Constant(2)
    z = x * y
    assert z.sample() == 2

def test_div():
    x = cs.Constant(1)
    y = cs.Constant(2)
    z = x / y
    assert z.sample() == 0.5

def test_mod():
    x = cs.Constant(1)
    y = cs.Constant(2)
    z = x % y
    assert z.sample() == 1

def test_floor_div():
    x = cs.Constant(1)
    y = cs.Constant(2)
    z = x // y
    assert z.sample() == 0

def test_order_of_operations():
    a = cs.Constant(2)
    b = cs.Constant(3)
    c = cs.Constant(4)
    d = cs.Constant(5)

    z = (a + b) * (c + d)

    assert z.sample() == (2 + 3) * (4 + 5)

    e = cs.Constant(6)

    z = (((a + b) * c) + d) * e

    assert z.sample() == (((2 + 3) * 4) + 5) * 6


def test_summation():
    T = cs.Constant(10)
    N = cs.Constant(5)

    som = cs.Summation(N, T)

    assert som.sample() == 50


def test_summation2():
    T = cs.Constant(10)
    N = cs.Constant(5)
    #     (50 - 30) * 20
    som = (cs.Summation(N, T) - cs.Constant(30)) * cs.Constant(20)

    assert som.sample() == 400


def test_summation_nested():
    T = cs.Summation(5,cs.Constant(1)*2) # 10
    N = cs.Summation(5,1) # 5

    som = (cs.Summation(N, T) - cs.Constant(30)) * cs.Constant(20)

    assert som.sample() == 400
