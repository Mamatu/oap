
#ifndef LHANDLE_H
#define	LHANDLE_H


#define BYTES_COUNT 8

/**
 * 
 */
class LHandle {
    int bytesCount;
    char byteRepresentation[BYTES_COUNT];
    void clear();
public:

    /**
     * Return pointer stored in this object.
     * Warning! If length of pointer which is stored in this class
     * is different than sizeof(void*) this method return NULL/
     *  @return Pointer which is stored in this object or NULL.
     */
    void* getPtr() const;

    LHandle();

    LHandle(void* ptr) ;

    LHandle(const LHandle& lHandle);

    LHandle& operator=(const LHandle& lhandle);

    bool operator==(const LHandle& lhandle);

    bool lessThan(const LHandle& lhandle) const;
};

bool operator<(const LHandle& handle1, const LHandle& handle2);
#endif	/* LHANDLE_H */

