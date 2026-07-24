
// g++ front.cpp -o name.exe -I "heder file locations" -L "lib file locations" -lSDL3
// final version


#include <sys/stat.h>
#define SDL_MAIN_USE_CALLBACKS 1  /* use the callbacks instead of main() */
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <functional>
#include <iostream>
#include "NN.hpp"
#include <vector>


// setting up the window and renderer for the game.
static SDL_Window *window = NULL;
static SDL_Renderer *renderer = NULL;


// Macro definitions
#define BOARD_SIZE 19
#define CELL_SIZE 36
#define MARGIN 30
#define WINDOW_SIZE (CELL_SIZE * (BOARD_SIZE - 1) + 2  * MARGIN)
#define STATUS_H 40
#define WINDOW_W  WINDOW_SIZE
#define WINDOW_H  (WINDOW_SIZE + STATUS_H)
#define PASS_MOVE (BOARD_SIZE * BOARD_SIZE)

constexpr int AI_MOVE_DELAY = 350;

enum controller {
    HUMAN,
    AI,
};

enum Screen {
    SCREEN_MENU,
    SCREEN_PLAY,
    SCREEN_OVER,
};

struct Appstate {
    Board board;
    NNGo model;
    controller control[2];
    Screen screen;
    int nextAIMove;
    const char *gameOverMessage;
};

// uses cell state enum to determine what color to draw for each cell and the turn of the player
// Enum for cell states

// pixel position of the board on the screen for the given grid coordinates of the board
static inline float boardX(int gx) { return (float)(MARGIN + gx * CELL_SIZE); }
static inline float boardY(int gy) { return (float)(MARGIN + gy * CELL_SIZE); }
static void drawtext(float x, float y, float scale, const char *str);
static void floodFillGroup(const Board &board, int x, int y, CellState color,std::vector<std::pair<int,int>>& stones,std::vector<std::pair<int,int>>& liberties,bool visited[BOARD_SIZE][BOARD_SIZE]) {
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
static void circlefill(float cx, float cy, float rad) {
    int r = (int)rad;
    for(int i = -r; i < r; i++){
        float x = SDL_sqrt(rad * rad - (float)(i * i));
        SDL_RenderLine(renderer, cx - x, cy - (float)i, cx + x, cy - (float)i);    
    }
    
}
// make the stone renderer


static void renderStone(Board *board, int gx, int gy, CellState color) {
    float cx = boardX(gx);
    float cy = boardY(gy);
    float rad = CELL_SIZE / 2.0f - 1.0f;
    
    SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
    circlefill(cx, cy, rad);

    //body
    if (color == BLACK_STONE) {
        SDL_SetRenderDrawColor(renderer, 40, 40, 40, 255);
    }
    else{
         SDL_SetRenderDrawColor(renderer, 230, 230, 230, 255);
    }
    circlefill(cx, cy, rad - 1.5f);
}

// make the board renderer 
void renderBoard(Appstate *app) {
    Board *board = &app->board;
    SDL_SetRenderDrawColor(renderer, 133, 94, 66, 255);
    SDL_RenderClear(renderer);

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);

    for (int i = 0; i < BOARD_SIZE; i++) {
        SDL_RenderLine(renderer, boardX(0), boardY(i), boardX(BOARD_SIZE - 1), boardY(i));
        SDL_RenderLine(renderer, boardX(i), boardY(0), boardX(i), boardY(BOARD_SIZE - 1));
    }

    //the star points
    const int star[3] = {3, 9, 15};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            circlefill(boardX(star[i]), boardY(star[j]), 3.0f);
        }
    }

    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (board->cells[i][j] != EMPTY) {
                renderStone(board, i, j, board->cells[i][j] == BLACK_STONE ? BLACK_STONE : WHITE_STONE);
            }
        }
    }
    // draw the status strip
    SDL_SetRenderDrawColor(renderer, 30, 24, 18, 255);
    SDL_FRect strip = { 0.0f, (float)WINDOW_W, (float)WINDOW_W, (float)STATUS_H };
    SDL_RenderFillRect(renderer, &strip);
    SDL_SetRenderDrawColor(renderer, 235, 235, 235, 255);

    char line[160];
    if (app->screen == SCREEN_OVER) {
        SDL_snprintf(line, sizeof(line), "%s   B:%d  W:%d   M=menu  ESC=quit",
                     app->gameOverMessage, board->score[0], board->score[1]);
    } else {
        SDL_snprintf(line, sizeof(line), "%s to move (%s)   B:%d  W:%d",board->turn == BLACK_STONE ? "Black" : "White",app->control[board->turn] == HUMAN ? "human" : "AI",board->score[0], board->score[1]);
    }
    drawtext(8.0f, (float)WINDOW_W + 10.0f, 1.5f, line);
    
};
    
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
static void drawtext(float x, float y, float scale, const char *str) {
    SDL_SetRenderScale(renderer, scale, scale);
    SDL_RenderDebugText(renderer, x / scale, y / scale, str);
    SDL_SetRenderScale(renderer, 1.0f, 1.0f);
}
static void renderMenu() {
    SDL_SetRenderDrawColor(renderer, 133, 94, 66, 255);
    SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer, 240, 240, 240, 255);

    drawtext(MARGIN, 80.0f, 3.5f, "GO Game (SDL3)");
    drawtext(MARGIN, 180.0f, 2.0f, "1: Human vs Human");
    drawtext(MARGIN, 220.0f, 2.0f, "2: Human (B) vs AI (W)");
    drawtext(MARGIN, 260.0f, 2.0f, "3: AI vs AI");

    drawtext(MARGIN, 380.0f, 1.8f, "Controls:");
    drawtext(MARGIN, 410.0f, 1.5f, "Click Intersections to Place Stone");
    drawtext(MARGIN, 435.0f, 1.5f, "P: Pass Turn | M: Menu | ESC: Quit");
}

static void resetBoard(Board &board) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            board.cells[i][j] = EMPTY;
        }
    }
    board.turn = BLACK_STONE;
    board.score[0] = 0;
    board.score[1] = 0;
    board.lastMove[0] = -1;
    board.lastMove[1] = -1;
    board.passes = 0;
}
/* This function runs once at startup. */
SDL_AppResult SDL_AppInit(void **appstate, int argc, char *argv[])
{
    Appstate *app = new Appstate();
    *appstate = app;

    if (!SDL_CreateWindowAndRenderer("Go Game", WINDOW_W, WINDOW_H, 0, &window, &renderer)) {
        SDL_Log("Couldn't create window and renderer: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    resetBoard(app->board);
    app->screen = SCREEN_MENU;
    app->control[0] = HUMAN;
    app->control[1] = HUMAN;
    app->nextAIMove = 0;

    SDL_Log("Game initialized successfully.");
    return SDL_APP_CONTINUE;
}



/* This function runs once per frame, and is the heart of the program. */
SDL_AppResult SDL_AppIterate(void *appstate) {
    Appstate *app = (Appstate *)appstate;

    if (app->screen == SCREEN_MENU) {
        renderMenu();
    } 
    else {
        // Handle AI Logic if it's the AI's turn
        if (app->screen == SCREEN_PLAY) {
            int turnIdx = (app->board.turn == BLACK_STONE) ? 0 : 1;
            if (app->control[turnIdx] == AI) {
                Uint64 now = SDL_GetTicks();
                if (now >= app->nextAIMove) {
                    // Compute legal moves mask
                    std::vector<bool> legalMoves(BOARD_SIZE * BOARD_SIZE, false);
                    for (int x = 0; x < BOARD_SIZE; ++x) {
                        for (int y = 0; y < BOARD_SIZE; ++y) {
                            if (app->board.cells[x][y] == EMPTY) {
                                legalMoves[x * BOARD_SIZE + y] = true;
                            }
                        }
                    }

                    // Query Neural Network Input
                    auto nnInput = boardToNNInput(app->board, app->board.turn);
                    
                    // Evaluate policy probabilities (using network output or mock equal policy if untrained)
                    std::vector<double> probs(BOARD_SIZE * BOARD_SIZE, 1.0 / (BOARD_SIZE * BOARD_SIZE));

                    int choice = sampleAction(probs, legalMoves);
                    if (choice == PASS_MOVE) {
                        app->board.passes++;
                        app->board.turn = (app->board.turn == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;
                        if (app->board.passes >= 2) {
                            CellState w = winner(app->board);
                            app->gameOverMessage = (w == BLACK_STONE) ? "Black Wins!" : "White Wins!";
                            app->screen = SCREEN_OVER;
                        }
                    } else {
                        int gx = choice / BOARD_SIZE;
                        int gy = choice % BOARD_SIZE;
                        int capts = 0, conns = 0;
                        makeMove(&app->board, gx, gy, capts, conns);
                    }
                    app->nextAIMove = SDL_GetTicks() + AI_MOVE_DELAY;
                }
            }
        }
        renderBoard(app);
    }

    SDL_RenderPresent(renderer);
    return SDL_APP_CONTINUE;
}
/* This function runs when a new event (mouse input, keypresses, etc) occurs. */
SDL_AppResult SDL_AppEvent(void *appstate, SDL_Event *event) {
    Appstate *app = (Appstate *)appstate;

    if (event->type == SDL_EVENT_QUIT) {
        return SDL_APP_SUCCESS;
    }

    if (event->type == SDL_EVENT_KEY_DOWN) {
        if (event->key.key == SDLK_ESCAPE) {
            return SDL_APP_SUCCESS;
        }

        if (event->key.key == SDLK_M) {
            app->screen = SCREEN_MENU;
            return SDL_APP_CONTINUE;
        }

        if (app->screen == SCREEN_MENU) {
            if (event->key.key == SDLK_1) {
                app->control[0] = HUMAN;
                app->control[1] = HUMAN;
                resetBoard(app->board);
                app->screen = SCREEN_PLAY;
            } else if (event->key.key == SDLK_2) {
                app->control[0] = HUMAN;
                app->control[1] = AI;
                resetBoard(app->board);
                app->screen = SCREEN_PLAY;
            } else if (event->key.key == SDLK_3) {
                app->control[0] = AI;
                app->control[1] = AI;
                resetBoard(app->board);
                app->screen = SCREEN_PLAY;
                app->nextAIMove = SDL_GetTicks() + AI_MOVE_DELAY;
            }
        } else if (app->screen == SCREEN_PLAY) {
            if (event->key.key == SDLK_P) {
                app->board.passes++;
                app->board.turn = (app->board.turn == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;
                if (app->board.passes >= 2) {
                    CellState w = winner(app->board);
                    app->gameOverMessage = (w == BLACK_STONE) ? "Black Wins!" : "White Wins!";
                    app->screen = SCREEN_OVER;
                }
            }
        }
    }

    if (app->screen == SCREEN_PLAY && event->type == SDL_EVENT_MOUSE_BUTTON_DOWN && event->button.button == SDL_BUTTON_LEFT) {
        int turnIdx = (app->board.turn == BLACK_STONE) ? 0 : 1;
        if (app->control[turnIdx] == HUMAN) {
            int gx = (int)SDL_lroundf((event->button.x - (float)MARGIN) / (float)CELL_SIZE);
            int gy = (int)SDL_lroundf((event->button.y - (float)MARGIN) / (float)CELL_SIZE);

            if (gx >= 0 && gx < BOARD_SIZE && gy >= 0 && gy < BOARD_SIZE) {
                int capts = 0, conns = 0;
                if (makeMove(&app->board, gx, gy, capts, conns)) {
                    app->nextAIMove = SDL_GetTicks() + AI_MOVE_DELAY;
                }
            }
        }
    }

    return SDL_APP_CONTINUE;
}

/* This function runs once at shutdown. */
void SDL_AppQuit(void *appstate, SDL_AppResult result)
{
    // removes the renderer and window, freeing up resources. at the end of the program.
    delete(Board *)appstate;
    if (renderer) {
        SDL_DestroyRenderer(renderer);
        renderer = NULL;
    }
    if (window) {
        SDL_DestroyWindow(window);
        window = NULL;
    }
    
}

