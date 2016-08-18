
#ifndef REFERENCESCOUNTER_H
#define	REFERENCESCOUNTER_H

#include <stdio.h>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>

template<typename T> class ReferencesCounter {
public:
    ReferencesCounter(const std::string& moduleName);
    virtual ~ReferencesCounter();

    T create();
    void destroy(T& t);

protected:
    virtual T createObject() = 0;
    virtual void destroyObject(T& t) = 0;

private:
    ReferencesCounter(const ReferencesCounter& orig);
    void checkReferences();

    std::string m_moduleName;
    typedef std::vector<T> References;
    References m_references;
};

template<typename T> ReferencesCounter<T>::ReferencesCounter(const std::string& moduleName) :
m_moduleName(moduleName) {
}

template<typename T> ReferencesCounter<T>::ReferencesCounter(const ReferencesCounter& orig) {
}

template<typename T> ReferencesCounter<T>::~ReferencesCounter() {
    checkReferences();
}

template<typename T> T ReferencesCounter<T>::create() {
    T t = createObject();
    m_references.push_back(t);
    return t;
}

template<typename T> void ReferencesCounter<T>::destroy(T& t) {
    typename References::iterator it = std::find(m_references.begin(), m_references.end(), t);
    if (it != m_references.end()) {
        m_references.erase(it);
    } else {
        fprintf(stderr, "\n\n [REFERENCE COUNTER] %s deallocation attempt. \n\n", m_moduleName.c_str());
    }
    destroyObject(t);
}

template<typename T> void ReferencesCounter<T>::checkReferences() {
    if (!m_references.empty()) {
        fprintf(stderr, "\n\n [REFERENCE COUNTER] %s detected memory leak. \n\n", m_moduleName.c_str());
        std::stringstream sstream;
        for (size_t fa = 0; fa < m_references.size(); ++fa) {
            sstream.str("");
            sstream << m_references[fa];
            fprintf(stderr, "\n\n [REFERENCE COUNTER] %s %s \n\n", m_moduleName.c_str(), sstream.str().c_str());
        }
    }
}

#endif	/* REFERENCESCOUNTER_H */

