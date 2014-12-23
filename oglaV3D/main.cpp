/* 
 * File:   main.cpp
 * Author: mmatula
 *
 * Created on September 14, 2014, 9:40 PM
 */

#include <cstdlib>

#include <SDL/SDL.h>
#include <GL/gl.h>
#include <GL/glu.h>

using namespace std;

bool handlePendingEvents() {
    SDL_Event keyEvent;
    while (SDL_PollEvent(&keyEvent) == 1) {
        if (keyEvent.type == SDL_KEYDOWN
                && keyEvent.key.keysym.sym == SDLK_ESCAPE) {
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    const SDL_VideoInfo* info = NULL;
    int width = 0;
    int height = 0;
    int bpp = 0;
    int flags = SDL_RESIZABLE;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        /* Failed, exit. */
        fprintf(stderr, "Video initialization failed: %s\n",
                SDL_GetError());
    }

    info = SDL_GetVideoInfo();

    if (!info) {
        /* This should probably never happen. */
        fprintf(stderr, "Video query failed: %s\n",
                SDL_GetError());

    }

    width = 640;
    height = 480;
    bpp = info->vfmt->BitsPerPixel;

    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 5);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 5);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 5);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    flags = SDL_OPENGL;
    if (SDL_SetVideoMode(width, height, bpp, flags) == 0) {
        fprintf(stderr, "Video mode set failed: %s\n",
                SDL_GetError());
    }
    while (handlePendingEvents()) {
    }
    return 0;
}

