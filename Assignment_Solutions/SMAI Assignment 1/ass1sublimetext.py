NumericalAttributes = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']
CategoricalAttributes = ['Work_accident', 'promotion_last_5years', 'sales', 'salary']

def BestOfAttributes(Attributes,examplesForCurAttr,targetAttribute,method):
    entropyForCurrentExamples = getEntropyOfExamples(examplesForCurAttr) 
    if len(Attributes) == 1:
        return Attributes[0]
    elif method == 'Entropy':
        EntropyForCategoricalAttributes = {}
        EntropyForNumbericalAttributes = {}
        for i in Attributes:
            if i in CategoricalAttributes:
                 EntropyForCategoricalAttributes[i] = getEntropyofAttribute(i,examplesForCurAttr,targetAttribute)
            elif i in NumericalAttributes:
                 EntropyForNumericalAttributes[i] = getEntropyofAttribute(i,examplesForCurAttr,targetAttribute)
        BestOfCategoricalAttributes = min(EntropyForCategoricalAttributes, key=EntropyForCategoricalAttributes.get)
        BestOfCategoricalAttributesValue = min(BestOfCategoricalAttributes.values())            
        BestOfNumericalAttributes = min(EntropyForNumericalAttributes, key=EntropyForNumericalAttributes.get)
        BestOfNumericalAttributesValue = min(BestOfNumericalAttributesValue.values())
        if len(Attributes) > 5:
            if BestOfCategoricalAttributesValue < BestOfNumericalAttributesValue:
                return BestOfCategoricalAttributes
            else
                return BestOfNumericalAttributes
            
                 
                
            return BestOfAttributes        
    elif method == 'Gini':
        GiniIndexForAttributes = {}
        for i in Attributes:
            GiniIndexForAttributes[i] = getGiniIndexForAttribute(i,examplesForCurAttr,targetAttribute)
        BestOfAttributes = min(GiniIndexForAttributes, key=GiniIndexForAttributes.get)
        return BestOfAttributes       
    elif method == 'misClasifi':
        MissClassiForAttributes = {}
        for i in Attributes:
            MissClassiForAttributes[i] = getMissClassiForAttribute(i,examplesForCurAttr,targetAttribute)
        BestOfAttributes = min(MissClassiForAttributes, key=MissClassiForAttributes.get)
        return BestOfAttributes    
    return BestOfAttributes







#(self, examples, attributes, decisionAttribute, decisionAttributeType, decisionAttributeValue, nodeNumber, parentDecisionAttribute, parentDecisionAttributeValue, decision, level):
def DecisionTreeBuilder(examplesForCurNode,targetAttribute,AttributesForCurrentNode,parentDecisionAttribute,parentAttrValue,level):
    if allExPos:
        r = Node(examplesForCurNode,AttributesForCurrentNode,'leafNode','leafNode','leafNode',numberOfNodes,parentDecisionAttribute,parentAttrValue,'Positive',level)
    if allExNeg:
        r = Node(examplesForCurNode,AttributesForCurrentNode,'leafNode','leafNode','leafNode',numberOfNodes,parentDecisionAttribute,parentAttrValue,'Negative',level)
    if len(AttributesForCurrentNode) == 0: #or #getEntropyOfExamples(examplesForCurNode) < 0.01:#or len(examplesForCurNode)<50 :
        print("entropy of current examples ")
        print(str(getEntropyOfExamples(examplesForCurNode)))
        fra1 = train.loc[examplesForCurNode,[targetAttribute] ]
        targetAttributesList = fra1[targetAttribute].values
        unique1, counts1 = np.unique(targetAttributesList, return_counts=True)
        dictOfTaAttr = dict(zip(unique1, counts1))
        posCount = dictOfTaAttr[1]
        negCount = dictOfTaAttr[0]
        if (posCount/(posCount+negCount))>0.24:
            r = Node(examplesForCurNode,AttributesForCurrentNode,'leafNode','leafNode','leafNode',numberOfNodes,parentDecisionAttribute,parentAttrValue,'Positive',level)
            numberOfNodes  = numberOfNodes + 1
        else:
            r = Node(examplesForCurNode,AttributesForCurrentNode,'leafNode','leafNode','leafNode',numberOfNodes,parentDecisionAttribute,parentAttrValue,'Negative',level)
            numberOfNodes  = numberOfNodes + 1
    BestAttribute,BestAttributeValue = BestOfAttributes(AttributesForCurrentNode,examplesForCurNode,targetAttribute,'misClasifi')
    if BestAttribute in NumericalAttributes:
        TypeOfdecisionAttribute = 'numerical'
    else:
        TypeOfdecisionAttribute = 'categorical'
        
    if TypeOfdecisionAttribute == 'categorical':
        r = Node(examplesForCurNode,AttributesForCurrentNode,BestAttribute,TypeOfdecisionAttribute,BestAttributeValue,numberOfNodes,parentDecisionAttribute,parentAttrValue,'NotLeaf',level)
        numberOfNodes  = numberOfNodes + 1
        BestAttributeValues = list(data[BestAttribute].unique())
        for iter in BestAttributeValues:
            AttributesForChildren = []
            AttributesForChildren = AttributesForCurrentNode[:]
            AttributesForChildren.remove(BestAttribute)
            #AttributesForChildren = AttributesForChildren + NumericalAttributes
            examplesForThisChildOfCurrentNode = []
            for iter1 in examplesForCurNode:
                if  str(train.at[int(iter1),str(BestAttribute)]) == str(iter):
                    examplesForThisChildOfCurrentNode.append(int(iter1))
            print("examplesForThisChildOfCurrentNode is "+str(len(examplesForThisChildOfCurrentNode)))
            if len(examplesForThisChildOfCurrentNode)==0:
                numberOfNodes  = numberOfNodes + 1
                fra1 = train.loc[examplesForCurNode,[targetAttribute] ]
                targetAttributesList = fra1[targetAttribute].values
                unique1, counts1 = np.unique(targetAttributesList, return_counts=True)
                dictOfTaAttr = dict(zip(unique1, counts1))
                posCount = dictOfTaAttr[1]
                negCount = dictOfTaAttr[0]
                if (posCount/(posCount+negCount))>0.24:
                    r.add_child(Node(examplesForThisChildOfCurrentNode,AttributesForCurrentNode,'leafNode','leafNode','leafNode',numberOfNodes,BestAttribute,str(iter),'Positive',level+1))                
                else:
                    r.add_child(Node(examplesForThisChildOfCurrentNode,AttributesForCurrentNode,'leafNode','leafNode','leafNode',numberOfNodes,BestAttribute,str(iter),'Negative',level+1))                    
                numberOfNodes = numberOfNodes - 1
            else:        
                r.add_child(DecisionTreeBuilder(examplesForThisChildOfCurrentNode,targetAttribute,AttributesForChildren,BestAttribute,str(iter),level+1))
        print("********************************************************************")
        print("The label of current node is")
        print(r)
        #print("the best decision attribute of current node is")
        #print(BestAttribute)
        #print("number of examplesForCurNode of this attribute is ")
        #print(len(examplesForCurNode))
        #print(examplesForCurNode)
        #print("children of current nodes")
        plabo = range(0,len(r.children))
        for k in plabo:
            pass
            #print(r.children[k])
        print("Type4")
        print("********************************************************************")

        return r
    else:
        r = Node(examplesForCurNode,AttributesForCurrentNode,BestAttribute,TypeOfdecisionAttribute,BestAttributeValue,numberOfNodes,parentDecisionAttribute,parentAttrValue,'NotLeaf',level)
        numberOfNodes  = numberOfNodes + 1
        #TheBestAttributeValues = [str(less than)+str(BestAttributeValue),str(greater than)+str(BestAttributeValue)]
        #for iter in TheBestAttributeValues:
        AttributesForChildren = []
        AttributesForChildren = AttributesForCurrentNode[:]
        examplesForLeftChildOfCurrentNode = []
        examplesForRightChildOfCurrentNode = []
        for iter1 in examplesForCurNode:
            if  float(str(train.at[int(iter1),str(BestAttribute)])) <= float(BestAttributeValue):
                examplesForLeftChildOfCurrentNode.append(int(iter1))
            else:
                examplesForRightChildOfCurrentNode.append(int(iter1))
        r.add_child(DecisionTreeBuilder(examplesForLeftChildOfCurrentNode,targetAttribute,AttributesForChildren,BestAttribute,BestAttributeValue,level+1))
        r.add_child(DecisionTreeBuilder(examplesForRightChildOfCurrentNode,targetAttribute,AttributesForChildren,BestAttribute,BestAttributeValue,level+1))
        return r
    return r
                
                    
        
        
        
        
        
        
        


    
    
    
    
decisionAttribute == 'leafNode' if leaf or bestdecision at internal node
decisionAttributeType == 'categorical' or 'numerical' or 'leafNode' if leaf
decisionAttributeValue = 'leafNode' if leaf
    
def predict(i,r,TestSet):
    if r.decisionAttribute == 'leafNode':
        if r.decision == 'Positive':
            return '1'
        else:
            #print("I am here in leaft")
            return '0'
    else:
        if r.decisionAttributeType == 'categorical': 
            AttrValue = str(TestSet.at[int(i),str(r.decisionAttribute)])
            #print(TestSet.at[int(i),str(r.decisionAttribute)])
            for j in r.children:
                if str(j.parentDecisionAttributeValue) == str(AttrValue):
                    break
            #print(j.parentDecisionAttributeValue)
            #print(AttrValue)
            if str(j.parentDecisionAttributeValue) == str(AttrValue):
                pass
            else:
                print("there is a problem")
            return predict(i,j,TestSet)
        elif r.decisionAttributeType == 'numerical':
            AttrValue = float(str(TestSet.at[int(i),str(r.decisionAttribute)]))
            if AttrValue > r.decisionAttributeValue
                return predict(i,r.children[1],TestSet)
            else:
                return predict(i,r.children[0],TestSet)
                
            
            
            
        
    