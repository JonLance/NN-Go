
// g++ front.cpp -o name.exe -I "heder file locations" -L "lib file locations" -lSDL3

 // add a player to choss so that if player is -- to 0 then it is a person and should wat for in put and if it is ==1 then it is an ai.
#include <sys/stat.h>
#define SDL_MAIN_USE_CALLBACKS 1  /* use the callbacks instead of main() */
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <iostream>

// setting up the window and renderer for the game.
static SDL_Window *window = NULL;
static SDL_Renderer *renderer = NULL;


// Macro definitions
#define BOARD_SIZE 19
#define CELL_SIZE 36
#define MARGIN 30
#define WINDOW_SIZE (CELL_SIZE * (BOARD_SIZE - 1) + 2  * MARGIN)

// uses cell state enum to determine what color to draw for each cell and the turn of the player
// Enum for cell states
enum CellState {
    EMPTY = 0,
    BLACK_STONE = 1,
    WHITE_STONE = 2,
};

struct Board {
    CellState cells[BOARD_SIZE][BOARD_SIZE];
    CellState turn;
    int score[2];
    int lastMove[2];
    int passes;
};
// pixel position of the board on the screen for the given grid coordinates of the board
static inline float boardX(int gx) { return (float)(MARGIN + gx * CELL_SIZE); }
static inline float boardY(int gy) { return (float)(MARGIN + gy * CELL_SIZE); }

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
void renderBoard(Board *board) {
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

/* This function runs once at startup. */
SDL_AppResult SDL_AppInit(void **appstate, int argc, char *argv[])
{
    /* Create the window */
    if (!SDL_CreateWindowAndRenderer("Go", WINDOW_SIZE, WINDOW_SIZE, 0, &window, &renderer)) {
        SDL_Log("Couldn't create window and renderer: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    // initializes the board to be empty.
    Board *board = new Board();
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            board->cells[i][j] = EMPTY;
        }
    }
    board->turn = BLACK_STONE;
    board->score[0] = 0;
    board->score[1] = 0;
    board->lastMove[0] = -1;
    board->lastMove[1] = -1;
    board->passes = 0;
    
    *appstate = board;
    SDL_Log("Click to place a stone and p to pass and esc to quit!");


    return SDL_APP_CONTINUE;
}



/* This function runs once per frame, and is the heart of the program. */
SDL_AppResult SDL_AppIterate(void *appstate)
{
    Board *board = (Board *)appstate;
    renderBoard(board);
    SDL_RenderPresent(renderer); // renders the present frame to the screen.
    return SDL_APP_CONTINUE;
}
/* This function runs when a new event (mouse input, keypresses, etc) occurs. */
SDL_AppResult SDL_AppEvent(void *appstate, SDL_Event *event)
{
    Board *board = (Board *)appstate;

    if (event->type == SDL_EVENT_QUIT)
        return SDL_APP_SUCCESS;

    // keyboard: Esc quits, P passes (two passes in a row ends the game)
    if (event->type == SDL_EVENT_KEY_DOWN) {
        if (event->key.key == SDLK_ESCAPE)
            return SDL_APP_SUCCESS;
        if (event->key.key == SDLK_P) {
            board->passes++;
            board->turn = (board->turn == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;
            if (board->passes >= 2) {
                SDL_Log("Game over. Captures - Black: %d, White: %d",
                        board->score[0], board->score[1]);
                return SDL_APP_SUCCESS;
            }
        }
    }
    if (event->type == SDL_EVENT_MOUSE_BUTTON_DOWN && event->button.button == SDL_BUTTON_LEFT) {
   
           // snap to nearest intersection (original rounded the pixel offset
           // and *then* divided, which truncates instead of snapping)
           int gx = (int)SDL_lroundf((event->button.x - (float)MARGIN) / (float)CELL_SIZE);
           int gy = (int)SDL_lroundf((event->button.y - (float)MARGIN) / (float)CELL_SIZE);
   
           // original had this test INVERTED — it bailed out on every valid move
           if (gx < 0 || gx >= BOARD_SIZE || gy < 0 || gy >= BOARD_SIZE){
               return SDL_APP_CONTINUE;
           }
           if (board->cells[gx][gy] != EMPTY){
               return SDL_APP_CONTINUE;
           }

           CellState pl1 = board->turn;
           CellState opp = (pl1 == BLACK_STONE) ? WHITE_STONE : BLACK_STONE;
   
           board->cells[gx][gy] = pl1;
   
           int capts = 0;
           capts += removeGroup(board, gx + 1, gy, opp);
           capts += removeGroup(board, gx - 1, gy, opp);
           capts += removeGroup(board, gx, gy + 1, opp);
           capts += removeGroup(board, gx, gy - 1, opp);
   
           bool visited[BOARD_SIZE][BOARD_SIZE] = { false };
           if (capts == 0 && !groupLiberties(board, gx, gy, pl1, visited)) {
               board->cells[gx][gy] = EMPTY;    // suicide, illegal
               return SDL_APP_CONTINUE;
           }
   
           board->score[(pl1 == BLACK_STONE) ? 0 : 1] += capts;
           board->lastMove[0] = gx;
           board->lastMove[1] = gy;
           board->passes = 0;
           board->turn = opp;
   
           return SDL_APP_CONTINUE;   // was SDL_APP_SUCCESS — that QUITS the app
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
