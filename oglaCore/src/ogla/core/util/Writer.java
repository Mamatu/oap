package ogla.core.util;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Class which write variables as string of bytes.
 *
 * @author marcin
 */
public class Writer {

    private ByteBuffer word = ByteBuffer.allocate(4);
    private ByteBuffer doubleWord = ByteBuffer.allocate(8);
    private byte[] body = new byte[0];
    private int index = 0;
    public static int BIG_ENDIAN = 0;
    public static int LITTLE_ENDIAN = 1;
    private int endianness = 1;

    protected byte[] getBytesRef() {
        return this.body;
    }

    public void setEndianness(int e) {
        this.endianness = e;
        if (e != 0 && e != 1) {
            this.endianness = 0;
        }
    }

    public Writer() {
    }

    public Writer(int capacity) {
        this.body = new byte[capacity];
    }

    /**
     * Clear array of bytes which belong to this class.
     */
    public void clear() {
        this.index = 0;
    }

    /**
     * Get size of byte's array.
     *
     * @return size of byte's array
     */
    public int size() {
        return this.index;
    }

    /**
     * Return array of bytes.
     *
     * @return bytes
     */
    public byte[] getBytes() {
        return body;
    }

    private void extend(byte[] bytes, int length) {
        if (this.index + length <= this.body.length) {
            System.arraycopy(bytes, 0, this.body, this.index, bytes.length);
        } else {
            int extraLength = (this.index + length) - this.body.length;
            byte[] b = new byte[this.body.length + extraLength];
            System.arraycopy(body, 0, b, 0, body.length);
            System.arraycopy(bytes, 0, b, body.length, bytes.length);
            this.body = b;
        }
        this.index += length;
    }

    /**
     * Write string to array of bytes which belong to this class object.
     *
     * @param text
     */
    public void write(String text) {
        byte[] bytes = text.getBytes();
        this.extend(bytes, bytes.length);
    }

    /**
     * Write float to array of bytes which belong to this class object.
     *
     * @param f
     * @throws IOException
     */
    public void write(float f) {
        fromFloat(f, word.array());
        this.extend(word.array(), word.capacity());
    }

    /**
     * Write double to array of bytes which belong to this class object.
     *
     * @param f
     * @throws IOException
     */
    public void write(double f) {
        fromDouble(f, doubleWord.array());
        this.extend(doubleWord.array(), doubleWord.capacity());
    }

    /**
     * Write integer to array of bytes which belong to this class object.
     *
     * @param i
     * @throws IOException
     */
    public void write(int i) {
        fromInt(i, word.array());
        this.extend(word.array(), word.capacity());
    }

    /**
     * Write bytes to array of bytes which belong to this class object.
     *
     * @param bytes
     */
    public void write(byte[] bytes) {
        this.extend(bytes, bytes.length);
    }

    public void write(Writer writer) {
        byte[] bytes = writer.getBytes();
        this.extend(bytes, bytes.length);
    }

    public void write(ByteBuffer byteBuffer) {
        byte[] bytes = byteBuffer.array();
        this.extend(bytes, bytes.length);
    }

    /**
     * Write boolean to array of bytes which belong to this class object.
     *
     * @param bytes
     */
    public void write(boolean v) {
        fromBoolean(v, word.array());
        this.extend(word.array(), word.capacity());
    }

    /**
     * Write boolean to array.
     *
     * @param b written variable
     * @param array to which is saved variable
     */
    public void fromBoolean(boolean b, byte[] array) {
        array[0] = 0;
        array[1] = 0;
        array[2] = 0;
        array[3] = 0;
        if (this.endianness == Writer.BIG_ENDIAN) {
            if (b) {
                array[0] = 1;
            } else {
                array[0] = 0;
            }
        } else {
            if (b) {
                array[3] = 1;
            } else {
                array[3] = 0;
            }
        }
    }

    private void setEndianness(ByteBuffer buffer) {
        if (this.endianness == Writer.BIG_ENDIAN) {
            buffer.order(ByteOrder.BIG_ENDIAN);
        } else if (this.endianness == Writer.LITTLE_ENDIAN) {
            buffer.order(ByteOrder.LITTLE_ENDIAN);
        }
    }

    /**
     * Write integer to array.
     *
     * @param b written variable
     * @param array to which is saved variable
     */
    public void fromInt(int v, byte[] array) {
        ByteBuffer byteBuffer = ByteBuffer.allocate(4);
        setEndianness(byteBuffer);
        byteBuffer.putInt(v);
        for (int fa = 0; fa < 4; fa++) {
            array[fa] = byteBuffer.get(fa);
        }
    }

    /**
     * Write float to array.
     *
     * @param b written variable
     * @param array to which is saved variable
     */
    public void fromFloat(float v, byte[] array) {
        ByteBuffer byteBuffer = ByteBuffer.allocate(4);
        setEndianness(byteBuffer);
        byteBuffer.putFloat(v);
        for (int fa = 0; fa < 4; fa++) {
            array[fa] = byteBuffer.get(fa);
        }
    }

    /**
     * Write double to array.
     *
     * @param b written variable
     * @param array to which is saved variable
     */
    public void fromDouble(double v, byte[] array) {
        ByteBuffer byteBuffer = ByteBuffer.allocate(8);
        setEndianness(byteBuffer);
        byteBuffer.putDouble(v);
        for (int fa = 0; fa < 8; fa++) {
            array[fa] = byteBuffer.get(fa);
        }
    }
}
