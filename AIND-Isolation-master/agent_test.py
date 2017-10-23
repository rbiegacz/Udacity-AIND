"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest
import timeit
import isolation
import game_agent

from importlib import reload

TIME_LIMIT_MILLIS = 300

DEBUG = False

def PrintD(msg):
    if DEBUG:
        print(msg)

class IsolationTest_minimax(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.timeout = 150
        self.player1 = game_agent.MinimaxPlayer(search_depth=2, score_fn=game_agent.custom_score, timeout=self.timeout)
        self.player2 = game_agent.MinimaxPlayer()
        self.game = isolation.Board(self.player1, self.player2, 9, 9)

    #@unittest.skip("disabled for debugging purposes")
    def test_minimax(self):
        self.game.apply_move((2, 2))
        self.game.apply_move((2, 3))
        self.game.apply_move((2, 4))
        self.game.apply_move((3, 2))
        self.game.apply_move((3, 3))
        self.game.apply_move((3, 4))
        self.game.apply_move((3, 5))
        self.game.apply_move((4, 2))
        self.game.apply_move((4, 3))
        self.game.apply_move((4, 4))
        self.game.apply_move((5, 2))
        self.game.apply_move((5, 3))
        self.game.apply_move((5, 4))
        self.game.apply_move((5, 5))
        self.game.apply_move((5, 6))
        self.game.apply_move((6, 3))
        self.game.apply_move((6, 4))
        self.game.apply_move((7, 6))

        self.game.apply_move((3, 6))
        self.game.apply_move((5, 7))

        PrintD("\n")
        PrintD(self.game.to_string())
        assert (self.player1 == self.game.active_player)
        time_millis = lambda: 1000 * timeit.default_timer()
        move_start = time_millis()
        time_left = lambda: TIME_LIMIT_MILLIS - (time_millis() - move_start)
        move = self.player1.get_move(self.game, time_left)
        PrintD("Result - the best move {}".format(move))
        assert(move != (-1, -1))
        assert(move == (1, 5) or move == (1, 7) or move == (2, 8) or move == (4, 8))


class IsolationTest_aphabeta1(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.timeout = 10
        self.player1 = game_agent.AlphaBetaPlayer(search_depth=1, score_fn=game_agent.custom_score, timeout=self.timeout)
        self.player2 = game_agent.AlphaBetaPlayer()
        self.game = isolation.Board(self.player1, self.player2, 9, 9)

    #@unittest.skip("disabled for debugging purposes")
    def test_alphabeta(self):
        self.game.apply_move((3, 2))
        self.game.apply_move((3, 3))
        self.game.apply_move((4, 4))
        self.game.apply_move((4, 5))
        self.game.apply_move((5, 4))
        self.game.apply_move((5, 5))
        self.game.apply_move((6, 3))
        self.game.apply_move((6, 6))

        self.game.apply_move((2, 4))
        self.game.apply_move((3, 5))
        PrintD("\n")
        PrintD(self.game.to_string())
        assert (self.player1 == self.game.active_player)
        time_millis = lambda: 1000 * timeit.default_timer()
        move_start = time_millis()
        time_left = lambda: TIME_LIMIT_MILLIS - (time_millis() - move_start)
        move = self.player1.get_move(self.game, time_left)
        print("Result - the best move {}".format(move))
        assert(move != (-1, -1))
        assert(move == (4, 3) or move == (3, 6))

class IsolationTest_aphabeta2(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.timeout = 10
        self.player1 = game_agent.AlphaBetaPlayer(search_depth=1, score_fn=game_agent.custom_score, timeout=self.timeout)
        self.player2 = game_agent.AlphaBetaPlayer()
        self.game = isolation.Board(self.player1, self.player2, 9, 9)

    #@unittest.skip("disabled for debugging purposes")
    def test_alphabeta(self):
        self.game.apply_move((2, 3))
        self.game.apply_move((2, 5))
        self.game.apply_move((2, 6))
        self.game.apply_move((3, 1))
        self.game.apply_move((3, 4))
        self.game.apply_move((3, 5))
        self.game.apply_move((4, 2))
        self.game.apply_move((4, 3))
        self.game.apply_move((4, 4))
        self.game.apply_move((4, 5))
        self.game.apply_move((4, 6))
        self.game.apply_move((6, 3))
        self.game.apply_move((6, 4))
        self.game.apply_move((6, 5))

        self.game.apply_move((2, 2))
        self.game.apply_move((1, 2))
        PrintD("\n")
        PrintD(self.game.to_string())
        assert (self.player1 == self.game.active_player)
        time_millis = lambda: 1000 * timeit.default_timer()
        move_start = time_millis()
        time_left = lambda: TIME_LIMIT_MILLIS - (time_millis() - move_start)
        move = self.player1.get_move(self.game, time_left)
        print("Result - the best move {}".format(move))
        assert(move != (-1, -1))
        assert(move == (0, 1) or move == (4, 1) or move == (0, 3))

class IsolationTest_aphabeta3(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.timeout = 10
        self.player1 = game_agent.AlphaBetaPlayer(search_depth=2, score_fn=game_agent.custom_score, timeout=self.timeout)
        self.player2 = game_agent.AlphaBetaPlayer()
        self.game = isolation.Board(self.player1, self.player2, 9, 9)

    #@unittest.skip("disabled for debugging purposes")
    def test_alphabeta(self):
        self.game.apply_move((2, 4))
        self.game.apply_move((2, 6))

        self.game.apply_move((3, 2))
        self.game.apply_move((3, 3))
        self.game.apply_move((3, 4))
        self.game.apply_move((3, 5))
        self.game.apply_move((3, 6))

        self.game.apply_move((4, 1))
        self.game.apply_move((4, 4))

        self.game.apply_move((5, 4))
        self.game.apply_move((5, 5))
        self.game.apply_move((5, 6))
        self.game.apply_move((6, 2))
        self.game.apply_move((6, 4))


        self.game.apply_move((2, 5))
        self.game.apply_move((6, 5))
        PrintD("\n")
        PrintD(self.game.to_string())
        assert (self.player1 == self.game.active_player)
        time_millis = lambda: 1000 * timeit.default_timer()
        move_start = time_millis()
        time_left = lambda: TIME_LIMIT_MILLIS - (time_millis() - move_start)
        move = self.player1.get_move(self.game, time_left)
        print("Result - the best move {}".format(move))
        assert(move != (-1, -1))
        assert(move == (4, 6) or move == (3, 7))

class IsolationTest_aphabeta4(unittest.TestCase):
    """Unit tests for isolation agents"""
    search_depth = 2
    def setUp(self):
        reload(game_agent)
        self.timeout = 10
        self.player1 = game_agent.AlphaBetaPlayer(search_depth=self.search_depth, score_fn=game_agent.custom_score, timeout=self.timeout)
        self.player2 = game_agent.AlphaBetaPlayer()
        self.game = isolation.Board(self.player1, self.player2, 9, 9)

    #@unittest.skip("disabled for debugging purposes")
    def test_alphabeta(self):
        self.game.apply_move((2, 2))
        self.game.apply_move((2, 6))

        self.game.apply_move((3, 2))

        self.game.apply_move((4, 3))
        self.game.apply_move((4, 4))
        self.game.apply_move((4, 5))
        self.game.apply_move((4, 6))

        self.game.apply_move((5, 3))
        self.game.apply_move((5, 4))
        self.game.apply_move((6, 2))

        self.game.apply_move((2, 5))
        self.game.apply_move((4, 7))
        PrintD("\n")
        PrintD("Search Depth: {}\n".format(self.search_depth))
        PrintD(self.game.to_string())
        assert (self.player1 == self.game.active_player)
        time_millis = lambda: 1000 * timeit.default_timer()
        move_start = time_millis()
        time_left = lambda: TIME_LIMIT_MILLIS - (time_millis() - move_start)
        move = self.player1.get_move(self.game, time_left)
        print("Result - the best move {}".format(move))
        assert(move != (-1, -1))
        assert(move == (3, 3) or move == (1, 3))

if __name__ == '__main__':
    unittest.main()
