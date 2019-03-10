import chess
import h5py
import numpy as np
import chess.syzygy

################ Board class functions
# status() check for problems with board
# is_valid() similar
# mirror()
# board = chess.Board()
# board.legal_moves
# print(board)
# print(board.unicode())

################ BaseBoard class
# chess.BaseBoard(board_fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR')
#       use instead of Board()? Might be faster.

# set_piece_map(pieces)
#       Sets up the board from a dictionary of pieces by square index.





########### Parses and creates SAN representation of moves.
board = chess.Board()
print(board.san(chess.Move(chess.E2, chess.E4)))
#'e4'
print(board.parse_san('Nf3'))
#Move.from_uci('g1f3')
print(board.variation_san([chess.Move.from_uci(m) for m in ["e2e4", "e7e5", "g1f3"]]))
#'1. e4 e5 2. Nf3'


############### Parses and creates FENs, extended FENs and Shredder FENs.
print()
print('--------------- Parses and creates FENs, extended FENs and Shredder FENs.')
print(board.fen())
#'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
print(board.shredder_fen())
#'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAha - 0 1'
board = chess.Board("8/8/8/2k5/4K3/8/8/8 w - - 4 45")
print(board)
print(board.piece_at(chess.C5))
#Piece.from_symbol('k')


############### prope syzygy
