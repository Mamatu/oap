/*
 * File:   main.cpp
 * Author: M.Matula
 *
 * Created on March 22, 2013, 9:20 AM
 */

#include <cstdlib>
#include <stdio.h>
#include <string>
#include <iostream>
#include <string.h>
#include <ios>
#include <map>
#include <linux/fs.h>
using namespace std;


std::string cmd_in("in");
std::string cmd_out("out");
std::string cmd_pos("pos");
std::string cmd_block("bsize");
std::string cmd_replace("replace");


int gcn = 5;
std::string gcmds[] = {cmd_in, cmd_out, cmd_pos, cmd_block, cmd_replace};

void Split(std::map<std::string, std::string>& commands, const char* value) {
    std::string proxy(value);
    int index = proxy.find('=', 0);
    std::string v(proxy.begin() + index + 1, proxy.end());
    std::string k(proxy.begin(), proxy.begin() + index);
    fprintf(stderr, "v = %s \n", v.c_str());
    fprintf(stderr, "k = %s \n", k.c_str());
    commands[k] = v;
}

class File {
public:
    long int sizeInBytes;
    FILE* file;

    File(const char* path, const char* mode) {
        file = fopen(path, mode);
        if (file) {
            fseek(file, 0, SEEK_END);
            sizeInBytes = ftell(file);
            fseek(file, 0, SEEK_SET);
        }
    }

    ~File() {
        if (file) {
            fclose(file);
        }
    }
};

class FilesOperations {
    File* in;
    File* out;
public:

    FilesOperations(File* _in, File* _out) : in(_in), out(_out) {
    }

    void replace(int position, int bsize, char* new_bytes, int new_size) {
        char* pbytes = new char[in->sizeInBytes];
        fread(pbytes, position*bsize, 1, in->file);
        memcpy(pbytes + position * bsize, new_bytes, new_size);
        fseek(in->file, new_size, SEEK_CUR);
        fread(pbytes + position * bsize + new_size, in->sizeInBytes - (position * bsize + new_size), 1, in->file);
        fwrite(pbytes, in->sizeInBytes, 1, out->file);
    }
};

typedef int(*func_t)(File* in, File* out, std::map<std::string, std::string>& args);
typedef std::map<std::string, std::string> Args;
typedef std::map<std::string, func_t> Cmds;

bool is(std::string v, std::map<std::string, std::string>& args) {
    return (args.find(v) != args.end());
}

int Replace(File* in, File* out, std::map<std::string, std::string>& args) {
    FilesOperations fo(in, out);

    if (is(cmd_pos, args) && is(cmd_block, args) && is(cmd_replace, args)) {

        int position = atoi(args.find(cmd_pos)->second.c_str());

        int bsize = atoi(args.find(cmd_block)->second.c_str());

        int b = atoi(args.find(cmd_replace)->second.c_str());
        char* bytes = (char*) &b;

        int new_size = bsize;

        fo.replace(position, bsize, bytes, new_size);
        return 0;

    }
    return 1;
}

void Execute(File* in, File* out, std::map<std::string, func_t> cmds, std::map<std::string, std::string>& args) {
    for (Args::iterator it = args.begin(); it != args.end(); it++) {
        Cmds::iterator cmd = cmds.find(it->first);
        if (cmd != cmds.end()) {
            cmd->second(in, out, args);
        }
    }
}

int main(int argc, char** argv) {
  Args args;

    Cmds cmds;
    cmds[cmd_replace] = Replace;

    for (int fa = 1; fa < argc; fa++) {
        Split(args, argv[fa]);
        Args::iterator it = args.end();
        it--;
        bool is = false;
        for (unsigned int fb = 0; fb < gcn && is == false; fb++) {
            if (gcmds[fb] == it->first) {
                is = true;
            }
        }
        if (!is) {
            printf("Error: commands %s is incorrect for program;", it->first.c_str());
            return 0;
        }
    }

    Args::iterator it_in = args.find(cmd_in);
    Args::iterator it_out = args.find(cmd_out);

    if (it_in == args.end()) {
        printf("Error: input file was not specified\n");
        return 0;
    }

    if (it_out == args.end()) {
        printf("Error: output file was not specified\n");
        return 0;
    }

    File in(it_in->second.c_str(), "rb");
    File out(it_out->second.c_str(), "wb");

    Execute(&in, &out, cmds, args);

    
    return 0;
}