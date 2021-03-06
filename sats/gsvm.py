from jnius import JavaClass, MetaJavaClass, JavaMethod, JavaMultipleMethod, cast, autoclass

SizeBasedUniqueRandomXOR = autoclass(
    'org.spectrumauctions.sats.core.bidlang.xor.SizeBasedUniqueRandomXOR')
JavaUtilRNGSupplier = autoclass(
    'org.spectrumauctions.sats.core.util.random.JavaUtilRNGSupplier')
Random = autoclass('java.util.Random')
HashSet = autoclass('java.util.HashSet')
Bundle = autoclass(
    'org.marketdesignresearch.mechlib.core.Bundle')
BundleEntry = autoclass(
    'org.marketdesignresearch.mechlib.core.BundleEntry')
GSVMStandardMIP = autoclass(
    'org.spectrumauctions.sats.opt.model.gsvm.GSVMStandardMIP')

class _Gsvm(JavaClass, metaclass=MetaJavaClass):
    __javaclass__ = 'org/spectrumauctions/sats/core/model/gsvm/GlobalSynergyValueModel'

    # createNewPopulation = JavaMultipleMethod([
    #    '()Ljava/util/List;',
    #    '(J)Ljava/util/List;'])
    setNumberOfNationalBidders = JavaMethod('(I)V')
    setNumberOfRegionalBidders = JavaMethod('(I)V')
    createWorld = JavaMethod(
        '(Lorg/spectrumauctions/sats/core/util/random/RNGSupplier;)Lorg/spectrumauctions/sats/core/model/gsvm/GSVMWorld;')
    createPopulation = JavaMethod(
        '(Lorg/spectrumauctions/sats/core/model/World;Lorg/spectrumauctions/sats/core/util/random/RNGSupplier;)Ljava/util/List;')
    setLegacyGSVM = JavaMethod('(Z)V')

    def __init__(self, seed, number_of_national_bidders, number_of_regional_bidders, isLegacyGSVM=False):
        super().__init__()
        if seed:
            rng = JavaUtilRNGSupplier(seed)
        else:
            rng = JavaUtilRNGSupplier()

        self.population = {}
        self.goods = {}
        self.efficient_allocation = None
        self.setNumberOfNationalBidders(number_of_national_bidders)
        self.setNumberOfRegionalBidders(number_of_regional_bidders)
        self.setLegacyGSVM(isLegacyGSVM)
        self.isLegacy = isLegacyGSVM
        self.world = self.createWorld(rng)
        self._bidder_list = self.createPopulation(self.world, rng)

        # Store bidders
        bidderator = self._bidder_list.iterator()
        count = 0
        while bidderator.hasNext():
            bidder = bidderator.next()
            self.population[count] = bidder
            count += 1

        # Store goods
        goods_iterator = self.world.getLicenses().iterator()
        count = 0
        while goods_iterator.hasNext():
            good = goods_iterator.next()
            assert good.getLongId() == count
            count += 1
            self.goods[good.getLongId()] = good

    def get_model_name(self):
        return ('GSVM')

    def get_bidder_ids(self):
        return self.population.keys()

    def get_good_ids(self):
        return self.goods.keys()

    def calculate_value(self, bidder_id, goods_vector):
        assert len(goods_vector) == len(self.goods.keys())
        bidder = self.population[bidder_id]
        bundleEntries = HashSet()
        for i in range(len(goods_vector)):
            if goods_vector[i] == 1:
                bundleEntries.add(BundleEntry(self.goods[i], 1))
        bundle = Bundle(bundleEntries)
        return bidder.calculateValue(bundle).doubleValue()

    def get_goods_of_interest(self, bidder_id):
        bidder = self.population[bidder_id]
        goods_of_interest = []
        for good_id, good in self.goods.items():
            good_set = HashSet()
            good_set.add(good)
            bundle = Bundle.of(good_set)
            if bidder.getValue(bundle, True).doubleValue() > 0:
                goods_of_interest.append(good_id)
        return goods_of_interest

    def get_uniform_random_bids(self, bidder_id, number_of_bids, seed=None):
        bidder = self.population[bidder_id]
        goods = autoclass('java.util.ArrayList')()
        for good in self.goods.values(): goods.add(good)
        if seed:
            random = Random(seed)
        else:
            random = Random()

        bids = []
        for i in range(number_of_bids):
            bid = []
            bundle = bidder.getAllocationLimit().getUniformRandomBundle(random, goods)
            for good_id, good in self.goods.items():
                if (bundle.contains(good)):
                    bid.append(1)
                else:
                    bid.append(0)
            bid.append(bidder.getValue(bundle).doubleValue())
            bids.append(bid)
        return bids

    def get_random_bids(self, bidder_id, number_of_bids, seed=None, mean_bundle_size=9, standard_deviation_bundle_size=4.5):
        bidder = self.population[bidder_id]
        if seed:
            rng = JavaUtilRNGSupplier(seed)
        else:
            rng = JavaUtilRNGSupplier()
        valueFunction = cast('org.spectrumauctions.sats.core.bidlang.xor.SizeBasedUniqueRandomXOR',
                                bidder.getValueFunction(SizeBasedUniqueRandomXOR, rng))
        valueFunction.setDistribution(
            mean_bundle_size, standard_deviation_bundle_size)
        valueFunction.setIterations(number_of_bids)
        xorBidIterator = valueFunction.iterator()
        bids = []
        while (xorBidIterator.hasNext()):
            bundleValue = xorBidIterator.next()
            bid = []
            for good_id, good in self.goods.items():
                if (bundleValue.getBundle().contains(good)):
                    bid.append(1)
                else:
                    bid.append(0)
            bid.append(bundleValue.getAmount().doubleValue())
            bids.append(bid)
        return bids

    def get_efficient_allocation(self, display_output=True):
        if self.efficient_allocation:
            return self.efficient_allocation, sum([self.efficient_allocation[bidder_id]['value'] for bidder_id in self.efficient_allocation.keys()])

        mip = GSVMStandardMIP(self.world, self._bidder_list)
        mip.setDisplayOutput(display_output)

        allocation = mip.calculateAllocation()

        self.efficient_allocation = {}

        for bidder_id, bidder in self.population.items():
            self.efficient_allocation[bidder_id] = {}
            self.efficient_allocation[bidder_id]['good_ids'] = []
            bidder_allocation = allocation.allocationOf(bidder)
            good_iterator = bidder_allocation.getBundle().getSingleQuantityGoods().iterator()
            while good_iterator.hasNext():
                self.efficient_allocation[bidder_id]['good_ids'].append(good_iterator.next().getLongId())

            self.efficient_allocation[bidder_id]['value'] = bidder_allocation.getValue().doubleValue()

        return self.efficient_allocation, allocation.getTotalAllocationValue().doubleValue()
