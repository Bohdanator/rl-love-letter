from game.card import Card
import numpy as np

from utils import tuplepop


class Game:
    """A Love Letter 2-player game"""

    def __init__(
        self,
        deck=[], 
        hands=[(), ()], 
        discards=[(), ()], 
        actions=[(), ()], 
        alive=[True, True], 
        knows=[False, False],
        turn=0
    ):
        self._deck = deck
        self._hands = hands
        self._discards = discards
        self._actions = actions
        self.alive = alive
        self._knows = knows
        self.turn = turn
    
    def copy_game(self):
        return Game(
            self._deck[:],
            self._hands[:],
            self._discards[:],
            self._actions[:],
            self.alive[:],
            self._knows[:],
            self.turn
        )
    
    def reset(self, np_random):
        self._deck = []
        for c, k in enumerate(Card.counts):
            self._deck += k*[c+1]
        np_random.shuffle(self._deck)
        self._hands = [(), ()]
        self._discards = [(), ()]
        self._actions = [(), ()]
        self.alive = [True, True]
        self._knows = [False, False]
        self.turn = 0
        self.draw_card(0)
        self.draw_card(1)
        self.draw_card(0)
    
    def draw_card(self, p=None, override=False):
        p = p if p is not None else self.turn%2
        if len(self._deck) > 1 or override:
            self._hands[p] += (self._deck.pop(), )
        else:
            raise Exception("Can't draw the last card in the deck")
    
    @property
    def running(self):
        return len(self._deck) > 1 and self.alive[0] and self.alive[1]
    
    @property
    def player(self):
        return self.turn % 2
    
    @property
    def opponent(self):
        return (self.turn + 1) % 2
    
    def kill(self, player):
        self.alive[player] = False
    
    def action_possible(self, action):
        card, guess, target = action
        if not self.alive[self.player]: 
            return False
        if card not in self._hands[self.player]:
            return False
        if card == 1 and guess == 0:
            return False
        if card in [Card.king, Card.prince] and Card.countess in self._hands[self.player]:
            return False
        return True
    
    def action(self, action):
        if not self.action_possible(action):
            raise Exception("Action not possible")
        if not self.running:
            raise Exception("Game ended")
        card, guess, target = action
        # discarding the played card
        self._discards[self.player] += (card, )
        # if opponent knows my card and I play it, he loses knowledge
        card1, card2 = self._hands[self.player]
        if card1 == card2 or card1 == card:
            self._knows[self.opponent] = False
        self._hands[self.player] = tuplepop(self._hands[self.player], card)
        self._actions[self.player] += ((card, guess, self._hands[target][0], target), )
        actions = {
            Card.guard: self._guard_action,
            Card.priest: self._priest_action,
            Card.baron: self._baron_action,
            Card.handmaid: self._handmaid_action,
            Card.prince: self._prince_action,
            Card.king: self._king_action,
            Card.countess: self._countess_action,
            Card.princess: self._princess_action,
        }
        if (
            card in (Card.guard, Card.priest, Card.baron, Card.king) and
            len(self._actions[target]) > 0 and
            self._actions[target][-1][0] == Card.handmaid
        ):
            pass
        else:
            actions[card](card, guess, target)
        self.turn += 1
        if self.running:
            self.draw_card()
    
    def _guard_action(self, card, guess, target):
        if guess == self._hands[target][0]:
            self.kill(target)
    
    def _priest_action(self, card, guess, target):
        if target != self.player:
            self._knows[self.player] = True
    
    def _baron_action(self, card, guess, target):
        self._actions[self.player] += ((card, guess, target), )
        if self._hands[self.player][0] > self._hands[target][0]:
            self.kill(target)
        elif self._hands[self.player][0] < self._hands[target][0]:
            self.kill(self.player)
    
    def _handmaid_action(self, card, guess, target):
        pass

    def _prince_action(self, card, guess, target):
        if self._hands[target][0] == Card.princess:
            self.kill(target)
        else:
            self._discards[target] += (self._hands[target][0], )
            self._hands[target] = ()
            self.draw_card(target, override=True)
            self._knows[1-target] = False

    def _king_action(self, card, guess, target):
        t = self._hands[target]
        self._hands[target] = self._hands[self.player]
        self._hands[self.player] = t
        self._knows = [True, True]
    
    def _countess_action(self, card, guess, target):
        pass
    
    def _princess_action(self, card, guess, target):
        self.kill(self.player)

    @property
    def action_conversion(self):
        """Action space is Discrete(15)"""
        return [
            (1,2,self.opponent),
            (1,3,self.opponent),
            (1,4,self.opponent),
            (1,5,self.opponent),
            (1,6,self.opponent),
            (1,7,self.opponent),
            (1,8,self.opponent),
            (2,0,self.opponent),
            (3,0,self.opponent),
            (4,0,self.opponent),
            (5,0,self.opponent),
            (5,0,self.player),
            (6,0,self.player),
            (7,0,self.player),
            (8,0,self.player)
        ]

    def convert_env_action(self, action):
        return self.action_conversion[action]
    
    def generate_possible_actions(self):
        # returns indices of actions that involve given card + the card left over
        possible_actions = np.zeros(len(self.action_conversion), dtype=np.int8)
        for i, a in enumerate(self.action_conversion):
            if a[0] in self._hands[self.player]:
                possible_actions[i] = 1
        if possible_actions[13]:
            possible_actions[12] = 0
            possible_actions[11] = 0
            possible_actions[10] = 0
        return possible_actions
    
    def observe(self, player=None):
        """Return observation of the game from the `player` perspective."""
        player = self.player if player is None else player

        first = np.zeros(8)
        first[self._hands[player][0]-1] = 1
        second = np.zeros(8)
        if len(self._hands[player]) > 1:
            second[self._hands[player][1]-1] = 1
        probabilities = np.zeros(8)
        if self._knows[player]:
            probabilities[self._hands[1-player][0]-1] = 1
        else:
            for card in self._deck + list(self._hands[1 - player]):
                probabilities[card-1] += 1
        return np.concatenate((first, second, (probabilities/(len(self._deck)+1))))
    
    def evaluate(self):
        if self.running:
            raise Exception("Game still running")
        if not self.alive[0] or not self.alive[1]:
            return self.alive
        p1 = self._hands[0][0]
        p2 = self._hands[1][0]
        if p1 == p2:
            p1 = sum(self._discards[0])
            p2 = sum(self._discards[1])
            if p1 == p2:
                # probably impossible case
                return (True, True)
        return (p1 > p2, p2 > p1)
    
    def print(self, player=None):
        player = player if player is not None else self.player

        print("*"*30)
        print(f"Love Letter Game (player {self.player} on turn)")
        print(f"Alive: {self.alive}")
        print(f"Deck: {len(self._deck)}")
        print(f"Discards: {self._discards}")
        print(f"Hands: {self._hands[player]}")
