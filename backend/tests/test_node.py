import functools

from graphs.learning_graph.nodes.node import Node, node


class TestNodeConstructor:
    def test_stores_fn(self):
        def my_func():
            return 42

        n = Node(fn=my_func)
        assert n.fn is my_func

    def test_stores_prompt(self):
        def my_func():
            return 42

        n = Node(fn=my_func, prompt="some prompt")
        assert n.prompt == "some prompt"

    def test_prompt_default_none(self):
        def my_func():
            return 42

        n = Node(fn=my_func)
        assert n.prompt is None

    def test_update_wrapper_applied(self):
        def my_func():
            """My docstring."""
            ...

        n = Node(fn=my_func)
        assert n.__name__ == "my_func"
        assert n.__doc__ == "My docstring."

    def test_update_wrapper_wrapped_attr_preserved(self):
        # functools.update_wrapper sets __wrapped__
        def my_func():
            return 42

        n = Node(fn=my_func)
        assert n.__wrapped__ is my_func


class TestNodeCall:
    def test_calls_underlying_fn(self):
        called = False

        def my_func():
            nonlocal called
            called = True

        n = Node(fn=my_func)
        n()
        assert called is True

    def test_returns_underlying_fn_result(self):
        def my_func():
            return "result"

        n = Node(fn=my_func)
        assert n() == "result"

    def test_forwards_positional_args(self):
        def my_func(a, b):
            return a + b

        n = Node(fn=my_func)
        assert n(1, 2) == 3

    def test_forwards_keyword_args(self):
        def my_func(*, x, y):
            return x * y

        n = Node(fn=my_func)
        assert n(x=3, y=4) == 12

    def test_forwards_mixed_args(self):
        def my_func(a, b, *, c):
            return (a + b) * c

        n = Node(fn=my_func)
        assert n(1, 2, c=3) == 9


class TestNodeDecoratorNoPrompt:
    def test_returns_node_instance(self):
        @node()
        def my_func():
            return 42

        assert isinstance(my_func, Node)

    def test_prompt_is_none(self):
        @node()
        def my_func():
            return 42

        assert my_func.prompt is None

    def test_decorated_function_works(self):
        @node()
        def my_func():
            return 42

        assert my_func() == 42


class TestNodeDecoratorWithPrompt:
    def test_returns_node_instance(self):
        @node(prompt="Helpful prompt")
        def my_func():
            return 42

        assert isinstance(my_func, Node)

    def test_stores_prompt(self):
        @node(prompt="Helpful prompt")
        def my_func():
            return 42

        assert my_func.prompt == "Helpful prompt"

    def test_name_and_docstring_preserved(self):
        @node(prompt="Helpful prompt")
        def my_func():
            """My docstring."""
            return 42

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."

    def test_decorated_function_works(self):
        @node(prompt="Helpful prompt")
        def my_func(a, b):
            return a + b

        assert my_func(3, 4) == 7


class TestNodeCallable:
    def test_node_is_callable(self):
        def my_func():
            return 42

        n = Node(fn=my_func)
        assert callable(n)

    def test_decorated_function_is_callable(self):
        @node(prompt="test")
        def my_func():
            return 42

        assert callable(my_func)