/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.core.rpc;

import java.nio.ByteBuffer;

/**
 *
 * @author mmatula
 */
class DynamicBuffer {

    private byte[] bytes = null;
    private int position = 0;

    void add(byte[] bytes) {
        if (this.bytes == null) {
            this.bytes = new byte[bytes.length];
            System.arraycopy(bytes, 0, this.bytes, 0, bytes.length);
        }
        if (this.bytes.length - position < bytes.length) {
            byte[] nbytes = new byte[bytes.length + position];
            System.arraycopy(this.bytes, 0, nbytes, 0, position);
            this.bytes = nbytes;
        }
        if (this.bytes.length - position >= bytes.length) {
            System.arraycopy(bytes, 0, this.bytes, position, bytes.length);
        }
        this.position += bytes.length;
    }

    void cutTo(int position) {
        this.position = 0;
        System.arraycopy(this.bytes, position, this.bytes, 0, this.bytes.length - position);
    }

    int size() {
        return this.position;
    }

    byte[] toArray() {
        byte[] output = new byte[this.size()];
        System.arraycopy(this.bytes, 0, output, 0, this.size());
        return output;
    }

    byte[] getByteRef() {
        return this.bytes;
    }
}
