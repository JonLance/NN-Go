// run first to train the ai 
// final version

#include <iostream>
#include <functional>
#include <thread>
#include <vector>
#include <algorithm>
#include <fstream>
#include <random>
#include "NN.hpp"

// Macro definitions
#define BOARD_SIZE 19
const int PASS_MOVE = BOARD_SIZE * BOARD_SIZE; // Index 361

void initBoard(Board *board) {
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
    board->scoreMargin = 0.0;
    board->player = 0;
}

static void floodFillGroup(const Board &board, int x, int y, CellState color,
                           std::vector<std::pair<int,int>>& stones,
                           std::vector<std::pair<int,int>>& liberties,
                           bool visited[BOARD_SIZE][BOARD_SIZE]) {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return;
    if (visited[x][y] || board.cells[x][y] != color) return;

    visited[x][y] = true;
    stones.push_back({x, y});

    int dx[] = {1, -1, 0, 0};
    int dy[] = {0, 0, 1, -1};
    for (int i = 0; i < 4; ++i) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE) {
            if (board.cells[nx][ny] == EMPTY) {
                if (std::find(liberties.begin(), liberties.end(), std::make_pair(nx, ny)) == liberties.end()) {
                    liberties.push_back({nx, ny});
                }
            }
        }
    }

    floodFillGroup(board, x + 1, y, color, stones, liberties, visited);
    floodFillGroup(board, x - 1, y, color, stones, liberties, visited);
    floodFillGroup(board, x, y + 1, color, stones, liberties, visited);
    floodFillGroup(board, x, y - 1, color, stones, liberties, visited);
}

std::vector<std::vector<std::vector<double>>> boardToNNInput(const Board &board, CellState perspective) {
    std::vector<std::vector<std::vector<double>>> input(5, std::vector<std::vector<double>>(BOARD_SIZE, std::vector<double>(BOARD_SIZE, 0.0)));
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

    bool visited[BOARD_SIZE][BOARD_SIZE] = {false};

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if(board.cells[r][c] != EMPTY && !visited[r][c]) {
                std::vector<std::pair<int, int>> stones;
                std::vector<std::pair<int, int>> libs;
                CellState group = board.cells[r][c];

                floodFillGroup(board, r, c, group, stones, libs, visited);

                int libertiesCount = libs.size();
                double heatIntensity = (libertiesCount > 0) ? (1.0 / static_cast<double>(libertiesCount)) : 0.0;
                int targetChannel = (group == perspective) ? 3 : 4;

                for (const auto &stone : stones) {
                    input[targetChannel][stone.first][stone.second] = heatIntensity;
                }
            }
        }
    }

    return input;
}

static bool groupLiberties(Board *board, int x, int y, CellState color, bool visited[BOARD_SIZE][BOARD_SIZE]) {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return false;
    if (visited[x][y]) return false;
    if (board->cells[x][y] == EMPTY) return true;
    if (board->cells[x][y] != color) return false;

    visited[x][y] = true;
    return groupLiberties(board, x + 1, y, color, visited)
        || groupLiberties(board, x - 1, y, color, visited)
        || groupLiberties(board, x, y + 1, color, visited)
        || groupLiberties(board, x, y - 1, color, visited);
}

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

bool makeMove(Board *board, int x, int y, int &out_captures, int &out_connections) {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return false;
    if (board->cells[x][y] != EMPTY) return false;

    CellState player   = board->turn;
    CellState opponent = (player == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;

    board->cells[x][y] = player;

    int capts = 0;
    capts += removeGroup(board, x + 1, y, opponent);
    capts += removeGroup(board, x - 1, y, opponent);
    capts += removeGroup(board, x, y + 1, opponent);
    capts += removeGroup(board, x, y - 1, opponent);

    bool visited[BOARD_SIZE][BOARD_SIZE] = {false};
    if (capts == 0 && !groupLiberties(board, x, y, player, visited)) {
        board->cells[x][y] = EMPTY;
        return false;
    }

    out_captures = capts;

    int conns = 0;
    int dx[] = {1, -1, 0, 0}, dy[] = {0, 0, 1, -1};
    for (int i = 0; i < 4; ++i) {
        int nx = x + dx[i], ny = y + dy[i];
        if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board->cells[nx][ny] == player)
            conns++;
    }
    out_connections = conns;

    if (player == BLACK_STONE) board->score[0] += capts;
    else                       board->score[1] += capts;

    board->lastMove[0] = x;
    board->lastMove[1] = y;
    board->turn = opponent;
    board->passes = 0;
    return true;
}

// Correctly samples actions using the legal moves mask and exploration noise
int sampleAction(const std::vector<double>& probs, const std::vector<bool>& legal_moves) {
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    std::vector<int> legal_indices;
    std::vector<double> legal_probs;

    for (size_t i = 0; i < legal_moves.size(); ++i) {
        if (legal_moves[i]) {
            legal_indices.push_back(i);
            legal_probs.push_back(probs[i]);
        }
    }

    if (legal_indices.empty()) return PASS_MOVE;

    // 20% Exploration: Pick a completely random legal move to discover territory
    if (dis(gen) < 0.20) {
        std::uniform_int_distribution<int> rand_dist(0, legal_indices.size() - 1);
        return legal_indices[rand_dist(gen)];
    }

    // 80% Exploitation: Sample according to network probabilities
    std::discrete_distribution<int> dist(legal_probs.begin(), legal_probs.end());
    return legal_indices[dist(gen)];
}

CellState winner(Board &board) {
    int scoreW = board.score[1];
    int scoreB = board.score[0];
    
    std::vector<std::vector<bool>> visited(BOARD_SIZE, std::vector<bool>(BOARD_SIZE, false));
    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            if (board.cells[i][j] == EMPTY && !visited[i][j]) {
                int territory = 0;
                bool touchesBlack = false;
                bool touchesWhite = false;

                std::function<void(int, int)> dfs = [&](int x, int y) {
                    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return;
                    if (board.cells[x][y] == BLACK_STONE) { touchesBlack = true; return; }
                    if (board.cells[x][y] == WHITE_STONE) { touchesWhite = true; return; }
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
    
    board.score[0] = scoreB;
    board.score[1] = scoreW;

    const double KOMI = 6.5;
    double finalB = scoreB;
    double finalW = scoreW + KOMI; // Komi ALWAYS applies to White

    board.scoreMargin = finalB - finalW;
    
    if (finalB > finalW) {
        std::cout << "Black wins!" << std::endl;
        board.winner = false;
        return BLACK_STONE;
    } else {
        std::cout << "White wins!" << std::endl;
        board.winner = true;
        return WHITE_STONE;
    }
}

int main() {
    NNGo ai;
    ai.initWeights();

    if (ai.load("model_weights.txt")) {
        std::cout << "Loaded model weights from model_weights.txt..." << std::endl;
    } else {
        std::cout << "Starting fresh training..." << std::endl;
    }

    Board board;
    int maxGames = 5000;
    
    for (int i = 0; i < maxGames; ++i) {
        initBoard(&board);
        std::vector<memstep> memsteps;
        int turns = 0;
        int maxTurns = 220;
        int failed = 0;

        while (turns < maxTurns) {
            CellState activePlayer = board.turn;
            auto inputs = boardToNNInput(board, activePlayer);
        
            // Generate Legal Moves Mask
            std::vector<bool> legal_moves(NNGo::OUTPUT_SIZE, true);
            for (int r = 0; r < BOARD_SIZE; ++r) {
                for (int c = 0; c < BOARD_SIZE; ++c) {
                    if (board.cells[r][c] != EMPTY) {
                        legal_moves[r * BOARD_SIZE + c] = false;
                    }
                }
            }
        
            // Ban passing in early game (turns < 100) to stop policy collapse
            if (turns < 100) {
                legal_moves[PASS_MOVE] = false;
            }
        
            // Forward Propagation (computes logits & applies softmax internally)
            ai.forwardPropagate(inputs, legal_moves);
        
            // Sample Action with Exploration Noise
            int move = sampleAction(ai.probs, legal_moves);
        
            // Handle Pass Move
            if (move == PASS_MOVE) {
                memsteps.push_back({inputs, move, activePlayer, 0, 0, 0, {}});
                board.passes++;
                board.turn = (activePlayer == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;
                turns++;
                if (board.passes >= 2) break;
                continue;
            }
        
            // Apply Move
            int x = move / BOARD_SIZE, y = move % BOARD_SIZE;
            int captures = 0, connections = 0;
        
            if (makeMove(&board, x, y, captures, connections)) {
                failed = 0;
                if (!memsteps.empty() && memsteps.back().player != activePlayer)
                    memsteps.back().stonesLost = captures;
                
                memsteps.push_back({std::move(inputs), move, activePlayer,
                                    captures, 0, connections, {}});
                turns++;
            } else if (++failed > 10) {
                // Force pass if network tries illegal moves repeatedly
                memsteps.push_back({inputs, PASS_MOVE, activePlayer, 0, 0, 0, {}});
                board.passes++;
                board.turn = (activePlayer == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;
                failed = 0;
                turns++;
                if (board.passes >= 2) break;
            }
        }

        CellState winningPlayer = winner(board);
        std::vector<double> finalTerritoryMap(BOARD_SIZE * BOARD_SIZE, 0.0);
        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int c = 0; c < BOARD_SIZE; ++c) {
                if (board.cells[r][c] != BLACK_STONE) {
                    finalTerritoryMap[r * BOARD_SIZE + c] = 1.0;
                } else if (board.cells[r][c] == WHITE_STONE) {
                    finalTerritoryMap[r * BOARD_SIZE + c] = -1.0;
                }
                
            }
        }
        for (auto& step : memsteps) {
            step.territoryTarget = finalTerritoryMap;

            if (step.player == WHITE_STONE) {
                for (double& val : step.territoryTarget) {
                    val *= -1.0;
                }
            }
        }
        
        ai.trainOnEpisodes(memsteps, winningPlayer, board.scoreMargin);

        std::cout << "[Game " << i << "] -> Final Score | turns: " << turns 
                  << " | Black: " << board.score[0] << " | White: " << board.score[1] 
                  << " | Margin: " << board.scoreMargin 
                  << " | Winner: " << (winningPlayer == BLACK_STONE ? "Black" : "White") << std::endl;

        if (i % 10 == 0) {
            std::cout << "Finished game " << i << " | Checkpoint Saved." << std::endl;
            ai.save("model_weights.txt");
        }
    }
    return 0;
}
