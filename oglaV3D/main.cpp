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

/*
 * 
 */
int main(int argc, char** argv) {
    /* Information about the current video settings. */
    const SDL_VideoInfo* info = NULL;
    /* Dimensions of our window. */
    int width = 0;
    int height = 0;
    /* Color depth in bits of our window. */
    int bpp = 0;
    /* Flags we will pass into SDL_SetVideoMode. */
    int flags = 0;

    /* First, initialize SDL's video subsystem. */
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        /* Failed, exit. */
        fprintf(stderr, "Video initialization failed: %s\n",
                SDL_GetError());
    }

    /* Let's get some video information. */
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
    
    
    while(true){}
    
    return 0;
}

