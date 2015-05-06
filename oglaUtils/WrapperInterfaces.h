/* 
 * File:   Error.h
 * Author: marcin
 *
 * Created on 09 December 2012, 14:39
 */

#ifndef OGLA_WRAPPER_INTERFACES_H
#define	OGLA_WRAPPER_INTERFACES_H
#include <vector>
#include <map>
#include <string>
#include "Argument.h"
#include "Reader.h"
#include "Writer.h"
#include "Callbacks.h"
#include "ArrayTools.h"
namespace utils {

    class OglaObject;

    template<typename T> class Container {
    protected:
        virtual ~Container();
    public:
        Container();
        virtual T* get(int index) const = 0;
        virtual int getCount() const = 0;
    };

    template<typename T>Container<T>::Container() {
    }

    template<typename T>Container<T>::~Container() {
    }

    template<typename T> class DefaultContainerImpl : public Container<T> {
    public:
        DefaultContainerImpl();
        T* get(int index) const;
        int getCount() const;
        bool add(T* object);
        bool remove(T* object);
    protected:
        virtual ~DefaultContainerImpl();
    private:
        std::vector<T*> container;

    };

    template<typename T>DefaultContainerImpl<T>::DefaultContainerImpl() : Container<T>() {
    }

    template<typename T>DefaultContainerImpl<T>::~DefaultContainerImpl() {
    }

    template<typename T>T* DefaultContainerImpl<T>::get(int index) const {
        return container[index];
    }

    template<typename T>int DefaultContainerImpl<T>::getCount() const {
        return container.size();
    }

    template<typename T>bool DefaultContainerImpl<T>::add(T* object) {
        int index = 1;
        GET_INDEX(typename std::vector<T*>, container, object, index);
        if (index == -1) {
            return false;
        }
        container.push_back(object);
        return true;
    }

    template<typename T>bool DefaultContainerImpl<T>::remove(T* object) {
        typename std::vector<T*>::iterator it;
        GET_ITERATOR(container, object, it);
        if (it == container.end()) {
            return false;
        }
        container.push_back(object);
        return true;
    }

    template<typename T> class Leaf {
    public:
        Leaf(T* root);
        Leaf();
        virtual ~Leaf();
        T* getRoot() const;
    protected:
        void setRoot(T* root);
    private:
        T* root;
    };

    template<typename T> Leaf<T>::Leaf(T* _root) : root(_root) {
        if (this == root) {
            abort();
        }
    }

    template<typename T> Leaf<T>::Leaf() : root(NULL) {
    }

    template<typename T>Leaf<T>::~Leaf() {
        this->root = NULL;
    }

    template<typename T>void Leaf<T>::setRoot(T* root) {
        if (this == root) {
            abort();
        }
        this->root = root;
    }

    template<typename T> T* Leaf<T>::getRoot() const {
        return this->root;
    }

    class Identificator : public Leaf<OglaObject> {
    protected:
        Identificator(OglaObject* root = NULL);
        void setName(const std::string& name);
        void setName(const char* name);
    public:
        char* getLinkedName(const char* linker) const;
        void getLinkedName(std::string& buffer, const char* linker) const;
        Identificator(const char* name, OglaObject* root = NULL);
        virtual ~Identificator();
        const char* getName() const;
        virtual bool equals(Identificator& another) const;
        bool equals(const char* name) const;
    private:
        std::string name;
    };

    class Info : public Identificator {
    protected:
        Info();
    public:
        Info(const char* name);
        virtual ~Info();
    protected:
        virtual const char* getInfo() = 0;
    };

    typedef void (*Function_f)(utils::Reader& input, utils::Writer* output, void * ptr);

    class OglaFunction : public Identificator {
    private:
        utils::ArgumentType* inputArguments;
        utils::ArgumentType* outputArguments;
        int inputArgc;
        int outputArgc;

        class FunctionsList {
        public:
            FunctionsList();
            utils::sync::Mutex functionsMutex;
            std::vector<OglaFunction*> functions;
            utils::Callbacks callbacks;
        };
        static FunctionsList functionsList;
    protected:
        virtual void invoked(utils::Reader& input, utils::Writer* output) = 0;
        OglaFunction(const char* name, const utils::ArgumentType* inputArgs, int inputArgc, OglaObject* root = NULL, const utils::ArgumentType* outputArgs = NULL, int outputArgc = 0);
        OglaFunction(const char* name, const utils::ArgumentType* inputArgs, int inputArgc, const utils::ArgumentType* outputArgs, int outputArgc);
        virtual ~OglaFunction();
    public:
        static int EVENT_CREATE_FUNCTION;
        static int EVENT_DESTROY_FUNCTION;
        static LHandle RegisterCallback(Callback_f callback, void* ptr);
        static void UnegisterCallback(LHandle callbackID);
        void invoke(utils::Writer& input, utils::Reader& output);
        void invoke(utils::Reader& input, utils::Writer& output);
        void invoke(utils::Reader& input, utils::Reader& output);
        void invoke(utils::Writer& input, utils::Writer& output);
        void invoke(utils::Writer& input);
        void invoke(utils::Reader& input);
        int getInputArgc() const;
        ArgumentType getInputArgumentType(int index) const;
        const ArgumentType* getInputArgumentsTypes() const;
        int getOutputArgc() const;
        ArgumentType getOutputArgumentType(int index) const;
        const ArgumentType* getOutputArgumentsTypes() const;
        bool equals(OglaFunction& function) const;
        bool equals(const char* name, const utils::ArgumentType* inputArguments, int inputArgc) const;
        friend class OglaObject;
    };

    class FunctionProxy : public OglaFunction {
    protected:
        Function_f functionPtr;
        OglaFunction* function;
        void invoked(utils::Reader& input, utils::Writer* output);
        void* ptr;
    public:
        FunctionProxy(Function_f _function, const char* name, const utils::ArgumentType* inputArgs, int inputArgc,
                OglaObject* root = NULL,
                const utils::ArgumentType* outputArgs = NULL, int outputArgc = 0, void* ptr = NULL);

        FunctionProxy(OglaFunction* _function);
        FunctionProxy(OglaFunction* _function, const char* name);
        virtual ~FunctionProxy();
    };

    class FunctionsContainer : public DefaultContainerImpl<OglaFunction> {
    public:
        FunctionsContainer();
        virtual ~FunctionsContainer();
    };

    class ObjectsContainer : public DefaultContainerImpl<OglaObject> {
    public:
        ObjectsContainer();
        virtual ~ObjectsContainer();
    };

    class OglaObject : public Identificator {
    private:
        bool objectsChain;
        Container<OglaFunction>* functions;
        Container<OglaObject>* objects;
    protected:
        std::string name;
        virtual ~OglaObject();
        void setObjectsContainer(Container<OglaObject>* objects);
        void setFunctionsContainer(Container<OglaFunction>* function);
    public:
        static void ConvertToStrings(OglaObject* object, const char*** names, int& namesCount);
        static void ConvertToStrings(OglaObject* object, std::vector<std::string>& names);
        static OglaObject* CreateObjects(const char** names, int namesCount);
        static void DestroyObjects(OglaObject* object);
        Container<OglaObject>* getObjectsContainer() const;
        Container<OglaFunction>* getFunctionsContainer() const;
        OglaObject(const char* name, OglaObject* root = NULL);
        OglaObject(const char** names, int namesCount);
    };

    class DefaultFunctionsContainer : public DefaultContainerImpl<utils::OglaFunction> {
    public:
        virtual ~DefaultFunctionsContainer();
    };
}

#endif	/* ERROR_H */
