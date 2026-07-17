#include <iostream>
#include <functional>
#include <vector>
#include <algorithm>
#include <fstream>
#include <random>
#include "NN.hpp"

// fix the errors between the NN.hpp and traning.cpp

// Macro definitions
#define BOARD_SIZE 19


void  initBoard(Board *board) {
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            board->cells[r][c] = EMPTY;
        }
    }
    board->turn = BLACK_STONE;
    board->score[0] = 0; // black
    board->score[1] = 0; // white
    board->lastMove[0] = -1;
    board->lastMove[1] = -1;
    board->passes = 0;
    board->winner = false;
}
// converting the board to NN input
std::vector<std::vector<std::vector<double>>> boardToNNInput(const Board &board, CellState perspective) {
    // the 3 states: empty, own stone, opponent's stone
    std::vector<std::vector<std::vector<double>>> input(3, std::vector<std::vector<double>>(BOARD_SIZE, std::vector<double>(BOARD_SIZE, 0.0)));

    CellState opponent = (perspective == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if (board.cells[r][c] == perspective) {
                input[0][r][c] = 1.0;
            } else if (board.cells[r][c] == opponent) {
                input[1][r][c] = 1.0;
            } else {
                input[2][r][c] = 1.0;
            }
        }
    }
    return input;
}

static bool groupLiberties(Board *board, int x, int y, CellState color, bool visited[BOARD_SIZE][BOARD_SIZE]) {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return false;
    if (visited[x][y]) return false;
    if (board->cells[x][y] == EMPTY) return true;   // found a liberty
    if (board->cells[x][y] != color) return false;  // enemy stone, dead end

    visited[x][y] = true;
    return groupLiberties(board, x + 1, y, color, visited)
        || groupLiberties(board, x - 1, y, color, visited)
        || groupLiberties(board, x, y + 1, color, visited)
        || groupLiberties(board, x, y - 1, color, visited);
}
// remove the stones that are surrounded by the opponent's stones in a group
static int removeLiberties(Board *board, int x, int y, CellState color) {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return 0;
    if (board->cells[x][y] != color) return 0;
    board->cells[x][y] = EMPTY;
    int n = 1;
    n += removeLiberties(board, x + 1, y, color);
    n += removeLiberties(board, x - 1, y, color);
    n += removeLiberties(board, x, y + 1, color);
    n += removeLiberties(board, x, y - 1, color);
    return n;
}

static int removeGroup(Board *board, int x, int y, CellState oppColor) {
    if(x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return 0;
    if(board->cells[x][y] != oppColor) return 0;
    bool visited[BOARD_SIZE][BOARD_SIZE] = {false};
    if(!groupLiberties(board, x, y, oppColor, visited)) {
        return removeLiberties(board, x, y, oppColor);
    }
    return 0;
}


bool makeMove(Board *board, int x, int y, CellState color) {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return false;
    if (board->cells[x][y] != EMPTY) return false;

    CellState player = board->turn;
    CellState oppent = (player == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;

    board->cells[x][y] = player;
    int capts = 0;
    capts += removeGroup(board, x + 1, y, oppent);
    capts += removeGroup(board, x - 1, y, oppent);
    capts += removeGroup(board, x, y + 1, oppent);
    capts += removeGroup(board, x, y - 1, oppent);


    if (player == BLACK_STONE) board->score[0] += capts;
    else board->score[1] += capts;

    bool visited[BOARD_SIZE][BOARD_SIZE] = {false};
    if(!groupLiberties(board, x, y, player, visited)) {
        board->cells[x][y] = EMPTY;
        if(player == BLACK_STONE) board->score[0] -= capts;
        else board->score[1] -= capts;
        return false;
    }
    // Switch to the opponent's turn
    board->turn = oppent;
    board->passes = 0;
    return true;
}


int sampleAction(const Board* board, const std::vector<double>& probs) {
    std::vector<double> legal_probs(probs.size(), 0.0);
    bool has_legal = false;

    for (int i = 0; i < 19 * 19; ++i) {
        int r = i / 19;
        int c = i % 19;
        if (board->cells[r][c] == EMPTY) {
            legal_probs[i] = probs[i];
            has_legal = true;
        }
    }

    if (!has_legal) return -1; // -1 signifies a pass natively

    std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<int> dist(legal_probs.begin(), legal_probs.end());
    return dist(gen);
}
// scoring & territory tracking
CellState winner(Board &board) {
    /*
     * the winner is the player with the higher score
     * the score is calculated by
     * {
     *  territory = 1 point per connected empty cell
     *  captures = 1 point per captured opponent stone
     *  deadstones = 2 points per dead stone
     * }
     */

    int scoreW = board.score[1];
    int scoreB = board.score[0];
    // cordant traking for territory with flood fill
    std::vector<std::vector<bool>> visited(BOARD_SIZE, std::vector<bool>(BOARD_SIZE, false));
    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            if (board.cells[i][j] == EMPTY && !visited[i][j]) {
                int territory = 0;
                bool touchesBlack = false;
                bool touchesWhite = false;
                // finding the territory with flood fill of each player for scoring.
                // this should be add to the frontend for scoring at the end of the game.
                std::function<void(int, int)> dfs = [&](int x, int y) {
                    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE || visited[x][y]) return;

                    if (board.cells[x][y] == BLACK_STONE) {
                        touchesBlack = true;
                        return;
                    }
                    if (board.cells[x][y] == WHITE_STONE) {
                        touchesWhite = true;
                        return;
                    }
                    if (visited[x][y]) return;
                    visited[x][y] = true;
                    territory++;
                    dfs(x + 1, y);
                    dfs(x - 1, y);
                    dfs(x, y + 1);
                    dfs(x, y - 1);
                };
                dfs(i, j);
                if (!touchesBlack && touchesWhite) {
                    scoreW += territory;
                } else if (touchesBlack && !touchesWhite) {
                    scoreB += territory;
                }
            }
        }
    }


    if (scoreB > scoreW) {
        std::cout << "Black wins!" << std::endl;
        board.winner = false;
        return BLACK_STONE; // Correctly returns CellState instead of bool
    } else {
        std::cout << "White wins!" << std::endl;
        board.winner = true;
        return WHITE_STONE; // Correctly returns CellState instead of bool
    }
}

int main() {
    NNGo ai;
    ai.initWeights();

    // loading the pre-existing brain if available
    if (ai.load("model_weights.txt")) {
        std::cout << "Loaded model weights from model_weights.txt..." << std::endl;
    } else {
        std::cout << "Starting fresh training..." << std::endl;
    }

    Board board;
    int maxGames = 5000; // number of games to train at a time
    for (int i = 0; i < maxGames; ++i) {
        initBoard(&board);
        std::vector<memstep> memsteps;
        int turns = 0;
        int maxTurns = 220; // maximum number of turns per game to avoid infinite loops
        while (turns < maxTurns) {
            CellState activePlayer = board.turn;
            auto inputs = boardToNNInput(board, activePlayer);
            // forward propagation to get predictions
            ai.forwardPropagate(inputs);
            // predictions and sampling actions
            int move = sampleAction(&board, ai.probs);
            if (move == -1) {
                board.passes++;
                board.turn = (board.turn == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;
                if (board.passes >= 2) break;
                continue;
            }

            int x = move / BOARD_SIZE;
            int y = move % BOARD_SIZE;

            if (makeMove(&board, x, y, board.turn)) {
                memsteps.push_back({inputs, move, activePlayer});
                turns++;
            }
            if (turns++) {
                board.passes = 0;
                board.turn = (board.turn == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;
            }

        }

        CellState winningPlayer = winner(board);
        ai.trainOnEpisodes(memsteps, winningPlayer);

        if(i % 100 == 0) {
            std::cout << "Finished game " << i << " | Winner: " << (winningPlayer == BLACK_STONE ? "Black" : "White") << " | File Checkpoint Saved." << std::endl;
            ai.save("model_weights.txt");
        }

    }
    return 0;
}
