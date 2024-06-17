<script>
	// Array of statements
	const statements = [
	  "Reviewing Supreme Court Cases",
	  "Researching historical case law",
	  "Analyzing legal precedents",
	  "Studying constitutional amendments",
	  "Examining landmark decisions",
	  "Evaluating court opinions",
	  "Investigating legal doctrines",
	  "Understanding judicial interpretations",
	  "Exploring case summaries",
	  "Assessing legal outcomes"
	];

	// Function to generate a random number between 1 and 10
	function getRandomStatement() {
	  const randomIndex = Math.floor(Math.random() * 10);
	  return statements[randomIndex];
	}

	// Store the selected statement
	let selectedStatement = getRandomStatement();
  </script>


<div class="w-full mt-3">
	<div class="animate-pulse flex w-full">
		<div class="space-y-2 w-full">
			<div class="h-2 bg-gray-200 dark:bg-gray-600 rounded mr-14" />
			<div class="grid grid-cols-3 gap-4">
				<div class="h-2 bg-gray-200 dark:bg-gray-600 rounded col-span-1" />
				<div class="h-2 bg-gray-200 dark:bg-gray-600 rounded col-span-2" />
			</div>

			<!-- Display the selected statement in the middle -->
			<div class="text-center py-4 text-l font-semibold">
				{selectedStatement}
			</div>

			<div class="grid grid-cols-3 gap-4">
				<div class="h-2 bg-gray-200 dark:bg-gray-600 rounded col-span-2" />
				<div class="h-2 bg-gray-200 dark:bg-gray-600 rounded col-span-1" />
			</div>
			<div class="grid grid-cols-4 gap-4">
				<div class="h-2 bg-gray-200 dark:bg-gray-600 rounded col-span-1" />
				<div class="h-2 bg-gray-200 dark:bg-gray-600 rounded col-span-2" />
				<div class="h-2 bg-gray-200 dark:bg-gray-600 rounded col-span-1 mr-4" />
			</div>

			<div class="h-2 bg-gray-200 dark:bg-gray-600 rounded" />
		</div>
	</div>
</div>
